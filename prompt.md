Create a new HWLang project in design/project.

Eventually it will become be an AI accelerator chip inspired by the early TPU designs.
Look at https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf for reference for any details I haven't specified.

Let's start with a basic version first. It should consist of the following submodules:
* the matrix multiply unit, a systolic array that does bf8 multiplications and accumulates into f32
* an activation function post-processing unit, just ReLU is fine for now
* the right DMA engines to feed data into this unit and write results back
  * for now just model these as interfaces that point to the outside world with basic 1-cycle latency read/write enable, address ports

Eventually we'll have proper control driven by a scalar execution unit, for now just write a simple control module that can accept basic instructions (an enum) of the form "do a matric multiply between matrices A/B at memory addresses ... with strides ...". and emits the right control signals to the other units for proper execution. For now just basic matrix multiply is fine, through we'll expand this to things like batched multiplies and convolution later.

Make modules parametrizable (at compile time using generics) and configurable (at runtime trough the control instructions) where it makes sense and does not add too much complexity. Create submodules (for the different units), and create separate files of generic utility types and functions (eg. for floating point implementations).


Also do throughout testing, using pytest, somewhat similar to how the unit tests for the compiler work, and even more similar to how testing works in https://github.com/KarelPeeters/advent_of_fpga. Specifically, create a single project, and also use that manifest for the tests.
* test utility functions by just calling them from python directly
* test submodules individually where it makes sense
* compare with reference implementations (eg. numpy)

Start by creating a quick summary of the how the RTL language this compiler implements works, based on the existing unit tests and the project I linked earlier, save that summary as hwl_docs.md and refer back to and fix it when necessary).
