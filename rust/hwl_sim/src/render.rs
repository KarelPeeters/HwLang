use crate::bits::{get_bit, get_unsigned};
use crate::consts::{
    COLOR_SEPARATOR_STROKE, COLOR_STATUS_TEXT, COLOR_TEXT_MUTED, COLOR_TEXT_STRONG, COLOR_WAVE_SIGNAL_STROKE,
};
use crate::format::{WaveRadix, format_value_for_type_with_radix};
use crate::time::{draw_time_grid, time_to_x};
use eframe::egui::{self, Align2, FontId, Rect, Stroke, pos2, vec2};
use hwl_language::sim::recorder::{WaveChange, WaveSignalType};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum WaveRenderMode {
    Digital,
    Analog,
}

#[derive(Debug, Copy, Clone)]
pub struct WaveDisplayOptions {
    pub radix: WaveRadix,
    pub render_mode: WaveRenderMode,
}

impl Default for WaveDisplayOptions {
    fn default() -> Self {
        Self {
            radix: WaveRadix::Dec,
            render_mode: WaveRenderMode::Digital,
        }
    }
}

pub fn draw_waveform(
    painter: &egui::Painter,
    rect: Rect,
    changes: &[WaveChange],
    bit_offset: usize,
    ty: &WaveSignalType,
    display_options: WaveDisplayOptions,
    pixels_per_time: f32,
    time_view_start: f32,
    max_time: u64,
) {
    painter.rect_stroke(rect, 0.0, Stroke::new(1.0, COLOR_SEPARATOR_STROKE));
    draw_time_grid(painter, rect, pixels_per_time, time_view_start, max_time);

    if changes.is_empty() {
        return;
    }

    let visible_start = time_view_start;
    let visible_end = (time_view_start + rect.width() / pixels_per_time).min(max_time as f32);
    if visible_end <= visible_start {
        return;
    }

    if display_options.render_mode == WaveRenderMode::Analog {
        draw_analog_waveform(
            painter,
            rect,
            changes,
            bit_offset,
            ty,
            pixels_per_time,
            time_view_start,
            max_time,
        );
        return;
    }

    if ty.bit_len() == 1 {
        for (index, change) in changes.iter().enumerate() {
            let start_time = change.time as f32;
            let end_time = changes
                .get(index + 1)
                .map(|next| next.time as f32)
                .unwrap_or(max_time as f32);
            let segment_start = start_time.max(visible_start);
            let segment_end = end_time.min(visible_end);
            if segment_end > segment_start {
                let y = bit_y(rect, get_bit(&change.bits, bit_offset));
                painter.line_segment(
                    [
                        pos2(time_to_x(rect, segment_start, time_view_start, pixels_per_time), y),
                        pos2(time_to_x(rect, segment_end, time_view_start, pixels_per_time), y),
                    ],
                    Stroke::new(1.5, COLOR_WAVE_SIGNAL_STROKE),
                );
            }
            if let Some(next) = changes.get(index + 1) {
                let transition_time = next.time as f32;
                if transition_time >= visible_start && transition_time <= visible_end {
                    let x = time_to_x(rect, transition_time, time_view_start, pixels_per_time);
                    painter.line_segment(
                        [
                            pos2(x, bit_y(rect, get_bit(&change.bits, bit_offset))),
                            pos2(x, bit_y(rect, get_bit(&next.bits, bit_offset))),
                        ],
                        Stroke::new(1.5, COLOR_WAVE_SIGNAL_STROKE),
                    );
                }
            }
        }
    } else {
        for (index, change) in changes.iter().enumerate() {
            let start_time = change.time as f32;
            let end_time = changes
                .get(index + 1)
                .map(|next| next.time as f32)
                .unwrap_or(max_time as f32);
            let segment_start = start_time.max(visible_start);
            let segment_end = end_time.min(visible_end);
            if segment_end <= segment_start {
                continue;
            }
            draw_bus_segment(
                &painter,
                rect,
                segment_start,
                segment_end,
                &format_value_for_type_with_radix(&change.bits, bit_offset, ty, display_options.radix),
                pixels_per_time,
                time_view_start,
            );
        }
    }
}

fn draw_analog_waveform(
    painter: &egui::Painter,
    rect: Rect,
    changes: &[WaveChange],
    bit_offset: usize,
    ty: &WaveSignalType,
    pixels_per_time: f32,
    time_view_start: f32,
    max_time: u64,
) {
    let values = changes
        .iter()
        .map(|change| (change.time, numeric_value_for_type(&change.bits, bit_offset, ty)))
        .collect::<Vec<_>>();
    let Some((min_value, max_value)) = values.iter().fold(None, |range, (_, value)| match range {
        None => Some((*value, *value)),
        Some((min_value, max_value)) => Some((f64::min(min_value, *value), f64::max(max_value, *value))),
    }) else {
        return;
    };
    let value_span = (max_value - min_value).max(1.0);
    let value_y =
        |value: f64| rect.bottom() - 5.0 - (((value - min_value) / value_span) as f32) * (rect.height() - 10.0);
    let visible_start = time_view_start;
    let visible_end = (time_view_start + rect.width() / pixels_per_time).min(max_time as f32);
    for (index, (time, value)) in values.iter().copied().enumerate() {
        let start_time = time as f32;
        let end_time = values
            .get(index + 1)
            .map(|(next_time, _)| *next_time as f32)
            .unwrap_or(max_time as f32);
        let segment_start = start_time.max(visible_start);
        let segment_end = end_time.min(visible_end);
        if segment_end <= segment_start {
            continue;
        }
        let y = value_y(value);
        painter.line_segment(
            [
                pos2(time_to_x(rect, segment_start, time_view_start, pixels_per_time), y),
                pos2(time_to_x(rect, segment_end, time_view_start, pixels_per_time), y),
            ],
            Stroke::new(1.5, COLOR_WAVE_SIGNAL_STROKE),
        );
        if let Some((next_time, next_value)) = values.get(index + 1).copied() {
            let transition_time = next_time as f32;
            if transition_time >= visible_start && transition_time <= visible_end {
                let x = time_to_x(rect, transition_time, time_view_start, pixels_per_time);
                painter.line_segment(
                    [pos2(x, y), pos2(x, value_y(next_value))],
                    Stroke::new(1.5, COLOR_WAVE_SIGNAL_STROKE),
                );
            }
        }
    }
    painter.text(
        rect.left_top() + vec2(4.0, 2.0),
        Align2::LEFT_TOP,
        format!("{min_value:.0}..{max_value:.0}"),
        FontId::monospace(10.0),
        COLOR_TEXT_MUTED,
    );
}

fn numeric_value_for_type(bits: &[u8], bit_offset: usize, ty: &WaveSignalType) -> f64 {
    match ty {
        WaveSignalType::Bool => {
            if get_bit(bits, bit_offset) {
                1.0
            } else {
                0.0
            }
        }
        &WaveSignalType::Int { signed, width } => {
            if signed && width > 0 && width <= 127 && get_bit(bits, bit_offset + width - 1) {
                let value = get_unsigned(bits, bit_offset, width) as i128 - (1i128 << width);
                value as f64
            } else {
                get_unsigned(bits, bit_offset, width.min(128)) as f64
            }
        }
        _ => get_unsigned(bits, bit_offset, ty.bit_len().min(128)) as f64,
    }
}

fn bit_y(rect: Rect, value: bool) -> f32 {
    if value { rect.top() + 6.0 } else { rect.bottom() - 6.0 }
}

fn draw_bus_segment(
    painter: &egui::Painter,
    rect: Rect,
    start_time: f32,
    end_time: f32,
    label: &str,
    pixels_per_time: f32,
    time_view_start: f32,
) {
    let x0 = time_to_x(rect, start_time, time_view_start, pixels_per_time);
    let x1 = time_to_x(rect, end_time, time_view_start, pixels_per_time);
    let y0 = rect.top() + 5.0;
    let y1 = rect.bottom() - 5.0;
    painter.line_segment([pos2(x0, y0), pos2(x1, y0)], Stroke::new(1.0, COLOR_STATUS_TEXT));
    painter.line_segment([pos2(x0, y1), pos2(x1, y1)], Stroke::new(1.0, COLOR_STATUS_TEXT));
    painter.line_segment([pos2(x0, y0), pos2(x0 + 4.0, y1)], Stroke::new(1.0, COLOR_STATUS_TEXT));
    painter.line_segment([pos2(x0, y1), pos2(x0 + 4.0, y0)], Stroke::new(1.0, COLOR_STATUS_TEXT));

    let segment_width = x1 - x0;
    let estimated_text_width = label.len() as f32 * 7.0;
    if segment_width > estimated_text_width + 12.0 {
        let clip_rect = Rect::from_min_max(pos2(x0 + 4.0, rect.top()), pos2(x1 - 4.0, rect.bottom()));
        painter.with_clip_rect(clip_rect).text(
            pos2((x0 + x1) / 2.0, rect.center().y),
            Align2::CENTER_CENTER,
            label,
            FontId::monospace(11.0),
            COLOR_TEXT_STRONG,
        );
    }
}
