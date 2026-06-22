import math
from pathlib import Path

import hwl
import pytest

PROJECT_MANIFEST = Path(__file__).parent / "../../../../../design/tpu/hwl.toml"

_compile_cache = None


def _compile() -> hwl.Compile:
    global _compile_cache
    if _compile_cache is None:
        s = hwl.Source.new_from_manifest_path(str(PROJECT_MANIFEST))
        _compile_cache = s.compile()
    return _compile_cache


def _clear_cache():
    global _compile_cache
    _compile_cache = None


# ---- helpers ----

def _bf8(sign: bool, exp: int, mant: int) -> hwl.Value:
    c = _compile()
    return c.resolve("fp_util.BF8").new(sign=sign, exp=exp, mant=mant)


def _f32(sign: bool, exp: int, mant: int) -> hwl.Value:
    c = _compile()
    return c.resolve("fp_util.F32").new(sign=sign, exp=exp, mant=mant)


def _bf8_from_float(x: float) -> hwl.Value:
    if x == 0.0:
        return _bf8(False, 0, 0)
    sign = x < 0
    x = abs(x)
    exp_raw = int(math.floor(math.log2(x)))
    biased = exp_raw + 15
    if biased <= 0:
        return _bf8(sign, 0, 0)
    if biased >= 31:
        return _bf8(sign, 31, 0)
    mant_val = (x / (2**exp_raw) - 1.0) * 4
    mant = int(round(mant_val))
    if mant >= 4:
        mant = 0
        biased += 1
    if biased >= 31:
        return _bf8(sign, 31, 0)
    return _bf8(sign, biased, mant)


def _bf8_to_float(v: hwl.Value) -> float:
    sign = -1.0 if v.sign else 1.0
    e, m = int(v.exp), int(v.mant)
    if e == 0 and m == 0:
        return 0.0
    if e == 31:
        return float("nan") if m != 0 else sign * float("inf")
    return sign * (2.0 ** (e - 15)) * (1.0 + m / 4.0)


def _f32_from_float(x: float) -> hwl.Value:
    if x == 0.0:
        return _f32(False, 0, 0)
    sign = x < 0
    x = abs(x)
    if math.isinf(x):
        return _f32(sign, 255, 0)
    if math.isnan(x):
        return _f32(False, 255, 1)
    exp_raw = int(math.floor(math.log2(x)))
    biased = exp_raw + 127
    if biased <= 0:
        return _f32(sign, 0, 0)
    if biased >= 255:
        return _f32(sign, 255, 0)
    mant_val = (x / (2**exp_raw) - 1.0) * (1 << 23)
    mant = int(round(mant_val))
    if mant >= (1 << 23):
        mant = 0
        biased += 1
    if biased >= 255:
        return _f32(sign, 255, 0)
    return _f32(sign, biased, mant)


def _f32_to_float(v: hwl.Value) -> float:
    sign = -1.0 if v.sign else 1.0
    e, m = int(v.exp), int(v.mant)
    if e == 0 and m == 0:
        return 0.0
    if e == 255:
        return float("nan") if m != 0 else sign * float("inf")
    return sign * (2.0 ** (e - 127)) * (1.0 + m / float(1 << 23))


# ---- Generic Float struct tests ----

class TestFloatStruct:
    def test_zero(self):
        c = _compile()
        Float = c.resolve("fp_util.Float")
        z = Float(5, 2).zero()
        assert z.sign == False
        assert int(z.exp) == 0
        assert int(z.mant) == 0

    def test_bias(self):
        c = _compile()
        Float = c.resolve("fp_util.Float")
        assert Float(5, 2).bias() == 15
        assert Float(8, 23).bias() == 127

    def test_is_zero(self):
        z = _bf8(False, 0, 0)
        n = _bf8(False, 15, 0)
        assert z.is_zero() == True
        assert n.is_zero() == False

    def test_is_nan(self):
        assert _bf8(False, 31, 1).is_nan() == True
        assert _bf8(False, 31, 0).is_nan() == False

    def test_is_inf(self):
        assert _bf8(False, 31, 0).is_inf() == True
        assert _bf8(False, 15, 0).is_inf() == False

    def test_is_subnormal(self):
        assert _bf8(False, 0, 1).is_subnormal() == True
        assert _bf8(False, 0, 0).is_subnormal() == False


# ---- Free function tests ----

class TestBF8Mul:
    def test_one_times_one(self):
        c = _compile()
        mul = c.resolve("fp_util.bf8_mul")
        r = mul(_bf8(False, 15, 0), _bf8(False, 15, 0))
        assert int(r.exp) == 127
        assert int(r.mant) == 0

    def test_two_times_two(self):
        c = _compile()
        mul = c.resolve("fp_util.bf8_mul")
        r = mul(_bf8(False, 16, 0), _bf8(False, 16, 0))
        assert int(r.exp) == 129
        assert int(r.mant) == 0

    def test_by_zero(self):
        c = _compile()
        mul = c.resolve("fp_util.bf8_mul")
        r = mul(_bf8(False, 15, 0), _bf8(False, 0, 0))
        assert r.is_zero() == True

    def test_sign(self):
        c = _compile()
        mul = c.resolve("fp_util.bf8_mul")
        pos = _bf8(False, 15, 0)
        neg = _bf8(True, 15, 0)
        assert mul(pos, neg).sign == True
        assert mul(neg, neg).sign == False

    def test_numpy_comparison(self):
        c = _compile()
        mul = c.resolve("fp_util.bf8_mul")
        values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, -1.0, -2.0, 0.25, 7.0]
        for x in values:
            for y in values:
                a = _bf8_from_float(x)
                b = _bf8_from_float(y)
                r_hw = mul(a, b)
                hw_f = _f32_to_float(r_hw)
                expected = _bf8_to_float(a) * _bf8_to_float(b)
                if abs(expected) < 1e-37:
                    assert abs(hw_f) < 1e-37, f"mul({x},{y}): {hw_f}"
                else:
                    rel = abs(hw_f - expected) / max(abs(expected), 1e-37)
                    assert rel < 1e-6, f"mul({x},{y}): {hw_f} vs {expected}"


class TestF32Add:
    def test_add_zero(self):
        c = _compile()
        add = c.resolve("fp_util.f32_add")
        one = _f32(False, 127, 0)
        z = _f32(False, 0, 0)
        r = add(one, z)
        assert int(r.exp) == 127
        assert int(r.mant) == 0

    def test_one_plus_one(self):
        c = _compile()
        add = c.resolve("fp_util.f32_add")
        one = _f32(False, 127, 0)
        r = add(one, one)
        assert int(r.exp) == 128
        assert int(r.mant) == 0

    def test_opposite_signs(self):
        c = _compile()
        add = c.resolve("fp_util.f32_add")
        r = add(_f32(False, 127, 0), _f32(True, 127, 0))
        assert r.is_zero() == True

    def test_diff_exponents(self):
        c = _compile()
        add = c.resolve("fp_util.f32_add")
        four = _f32(False, 129, 0)   # 4.0
        one = _f32(False, 127, 0)    # 1.0
        r = add(four, one)
        assert int(r.exp) == 129
        assert int(r.mant) == (1 << 21)  # (5/4 - 1)*2^23

    def test_numpy_comparison(self):
        c = _compile()
        add = c.resolve("fp_util.f32_add")
        vals = [0.0, 0.5, 1.0, 2.0, 3.0, -1.0, -2.0, 10.0, 100.0, 1.5]
        for x in vals:
            for y in vals:
                a = _f32_from_float(x)
                b = _f32_from_float(y)
                r = add(a, b)
                hw_f = _f32_to_float(r)
                expected = x + y
                if abs(expected) < 1e-37:
                    assert abs(hw_f) < 1e-37, f"add({x},{y}): {hw_f}"
                else:
                    rel = abs(hw_f - expected) / max(abs(expected), 1e-37)
                    assert rel < 1e-5, f"add({x},{y}): {hw_f} vs {expected}"


class TestF32ReLU:
    def test_positive(self):
        c = _compile()
        relu = c.resolve("fp_util.f32_relu")
        r = relu(_f32_from_float(3.5))
        assert _f32_to_float(r) == pytest.approx(3.5)

    def test_negative(self):
        c = _compile()
        relu = c.resolve("fp_util.f32_relu")
        r = relu(_f32_from_float(-3.5))
        assert _f32_to_float(r) == 0.0

    def test_negative_zero(self):
        c = _compile()
        relu = c.resolve("fp_util.f32_relu")
        r = relu(_f32(True, 0, 0))
        assert r.sign == False  # -0 -> +0


class TestBF8ToF32:
    def test_one(self):
        c = _compile()
        conv = c.resolve("fp_util.bf8_to_f32")
        r = conv(_bf8(False, 15, 0))
        assert int(r.exp) == 127
        assert int(r.mant) == 0

    def test_zero(self):
        c = _compile()
        conv = c.resolve("fp_util.bf8_to_f32")
        r = conv(_bf8(False, 0, 0))
        assert r.is_zero() == True


# ---- Module tests ----

class TestSystolicPE:
    def test_resolves(self):
        pe = _compile().resolve("systolic.systolic_pe")
        assert isinstance(pe, hwl.Module)

    def test_verilog(self, tmp_dir: Path):
        pe = _compile().resolve("systolic.systolic_pe")
        assert "systolic_pe" in pe.as_verilog().source


class TestSystolicArray:
    def test_verilog(self, tmp_dir: Path):
        sa = _compile().resolve("systolic.systolic_array")
        m = sa(N=2)
        verilog = m.as_verilog().source
        assert "systolic_pe" in verilog
        assert "systolic_array" in verilog


class TestActivationUnit:
    def test_verilog(self, tmp_dir: Path):
        au = _compile().resolve("activation.activation_unit")
        verilog = au(N=4).as_verilog().source
        assert "activation_unit" in verilog


class TestDMAEngine:
    def test_verilog(self, tmp_dir: Path):
        dma = _compile().resolve("dma.dma_engine")
        verilog = dma(W=8).as_verilog().source
        assert "dma_engine" in verilog


class TestControlUnit:
    def test_insn_construct(self):
        c = _compile()
        Insn = c.resolve("control.Insn")
        Params = c.resolve("control.MatMulParams")
        p = Params.new(a_addr=0x1000, b_addr=0x2000, c_addr=0x3000,
                       M=16, N=16, K=16, stride_a=16, stride_b=16, stride_c=16)
        assert str(Insn.MatMul(p)).startswith("Insn.MatMul")

    def test_verilog(self, tmp_dir: Path):
        cu = _compile().resolve("control.control_unit")
        assert "control_unit" in cu.as_verilog().source


class TestAccelerator:
    def test_verilog(self, tmp_dir: Path):
        acc = _compile().resolve("top.accelerator")
        m = acc(ARRAY_SIZE=2, MEM_DATA_WIDTH=8)
        verilog = m.as_verilog().source
        for name in ["systolic_pe", "dma_engine", "control_unit", "activation_unit"]:
            assert name in verilog, f"missing {name}"
