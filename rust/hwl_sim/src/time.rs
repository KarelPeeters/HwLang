use crate::consts::{
    COLOR_CURSOR, COLOR_CURSOR_DELETE_BORDER, COLOR_CURSOR_STATS_BG, COLOR_CURSOR_STATS_BORDER, COLOR_TEXT_MUTED,
    COLOR_TEXT_STRONG, COLOR_TIME_GRID, COLOR_TIME_SCROLL_BORDER, COLOR_TIME_SCROLL_HANDLE_FILL,
    COLOR_TIME_SCROLL_HANDLE_STROKE, COLOR_TIME_SCROLL_TRACK, COLOR_ZOOM_SHADE, CURSOR_STATS_COLUMN_WIDTH,
    MAX_PIXELS_PER_TIME, MIN_PIXELS_PER_TIME,
};
use eframe::egui::{self, Align2, FontId, Rect, Sense, Stroke, Ui, pos2, vec2};
use hwl_language::sim::recorder::WaveChange;

#[derive(Debug, Copy, Clone)]
pub struct ZoomDrag {
    pub start_time: u64,
    pub current_time: u64,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct EdgeCounts {
    pub posedges: usize,
    pub negedges: usize,
    pub toggles: usize,
}

pub fn clamp_time_view_start(time_view_start: f32, visible_duration: f32, max_time: u64) -> f32 {
    let max_time = max_time as f32;
    if visible_duration >= max_time {
        0.0
    } else {
        time_view_start.clamp(0.0, max_time - visible_duration)
    }
}

pub fn zoom_to_selection(visible_wave_width: f32, max_time: u64, a: u64, b: u64) -> Option<(f32, f32)> {
    let start = a.min(b);
    let end = a.max(b);
    let duration = end.checked_sub(start)?;
    if duration == 0 {
        return None;
    }
    let pixels_per_time = (visible_wave_width / duration as f32).clamp(MIN_PIXELS_PER_TIME, MAX_PIXELS_PER_TIME);
    let visible_duration = visible_wave_width / pixels_per_time;
    let time_view_start = clamp_time_view_start(start as f32, visible_duration, max_time);
    Some((time_view_start, pixels_per_time))
}

pub fn wave_content_width(time_view_start: f32, pixels_per_time: f32, visible_width: f32, max_time: u64) -> f32 {
    ((max_time as f32 - time_view_start).max(0.0) * pixels_per_time)
        .min(visible_width)
        .max(1.0)
}

pub fn time_from_pointer(
    axis_rect: Rect,
    pointer_pos: egui::Pos2,
    pixels_per_time: f32,
    time_view_start: f32,
    max_time: u64,
) -> u64 {
    (time_view_start + (pointer_pos.x - axis_rect.left()) / pixels_per_time)
        .round()
        .clamp(0.0, max_time as f32) as u64
}

pub fn draw_time_view_range(
    ui: &mut Ui,
    label_width: f32,
    wave_width: f32,
    time_view_start: &mut f32,
    pixels_per_time: f32,
    max_time: u64,
) {
    let height = 34.0;
    let (row_rect, _) = ui.allocate_exact_size(vec2(label_width + wave_width, height), Sense::hover());
    let label_rect = Rect::from_min_size(row_rect.min, vec2(label_width, height));
    let track_rect = Rect::from_min_size(
        pos2(row_rect.left() + label_width, row_rect.center().y - 3.0),
        vec2(wave_width, 6.0),
    );
    let response = ui.interact(
        track_rect,
        ui.make_persistent_id("time-view-range"),
        Sense::click_and_drag(),
    );
    let visible_duration = wave_width / pixels_per_time;
    if (response.clicked() || response.dragged()) && max_time > 0 {
        if let Some(pos) = response.interact_pointer_pos() {
            let center_time = ((pos.x - track_rect.left()) / track_rect.width()).clamp(0.0, 1.0) * max_time as f32;
            *time_view_start = clamp_time_view_start(center_time - visible_duration / 2.0, visible_duration, max_time);
        }
    }

    let painter = ui.painter_at(row_rect);
    let max_time_f = max_time as f32;
    let visible_end = (*time_view_start + visible_duration).min(max_time_f);
    painter.text(
        pos2(label_rect.left(), label_rect.center().y),
        Align2::LEFT_CENTER,
        format!("Visible {:.0}..{:.0} / {max_time}", *time_view_start, visible_end),
        FontId::proportional(12.0),
        COLOR_TEXT_MUTED,
    );
    painter.rect_filled(track_rect, 3.0, COLOR_TIME_SCROLL_TRACK);
    painter.rect_stroke(track_rect, 3.0, Stroke::new(1.0, COLOR_TIME_SCROLL_BORDER));

    let handle_left = track_rect.left() + (*time_view_start / max_time_f) * track_rect.width();
    let handle_right = track_rect.left() + (visible_end / max_time_f) * track_rect.width();
    let handle_rect = Rect::from_min_max(
        pos2(handle_left, track_rect.center().y - 7.0),
        pos2(
            handle_right.max(handle_left + 8.0).min(track_rect.right()),
            track_rect.center().y + 7.0,
        ),
    );
    painter.rect_filled(handle_rect, 3.0, COLOR_TIME_SCROLL_HANDLE_FILL);
    painter.rect_stroke(handle_rect, 3.0, Stroke::new(1.0, COLOR_TIME_SCROLL_HANDLE_STROKE));
}

pub fn draw_time_axis(
    ui: &mut Ui,
    pixels_per_time: f32,
    time_view_start: f32,
    max_time: u64,
    label_width: f32,
    width: f32,
) -> Rect {
    let height = 24.0;
    let (row_rect, _) = ui.allocate_exact_size(vec2(label_width + width, height), Sense::hover());
    let content_width = wave_content_width(time_view_start, pixels_per_time, width, max_time);
    let rect = Rect::from_min_size(
        pos2(row_rect.left() + label_width, row_rect.top()),
        vec2(content_width, height),
    );
    let painter = ui.painter_at(rect);
    painter.line_segment(
        [pos2(rect.left(), rect.bottom()), pos2(rect.right(), rect.bottom())],
        Stroke::new(1.0, COLOR_TEXT_MUTED),
    );
    draw_time_grid(&painter, rect, pixels_per_time, time_view_start, max_time);

    let tick_step = major_tick_step(pixels_per_time);
    let mut t = first_tick_at_or_after(time_view_start, tick_step);
    while t <= max_time {
        let x = rect.left() + (t as f32 - time_view_start) * pixels_per_time;
        if x > rect.right() {
            break;
        }
        painter.line_segment(
            [pos2(x, rect.bottom()), pos2(x, rect.bottom() - 6.0)],
            Stroke::new(1.0, COLOR_TEXT_MUTED),
        );
        painter.text(
            pos2(x + 2.0, rect.top()),
            Align2::LEFT_TOP,
            t.to_string(),
            FontId::monospace(11.0),
            COLOR_TEXT_MUTED,
        );
        t = t.saturating_add(tick_step);
    }
    rect
}

pub fn draw_cursor_stats_header(ui: &mut Ui, axis_rect: Rect, cursor_time: u64, secondary_cursor_time: u64) -> bool {
    let rect = Rect::from_min_max(
        pos2(axis_rect.left() - CURSOR_STATS_COLUMN_WIDTH, axis_rect.top()),
        pos2(axis_rect.left(), axis_rect.bottom()),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, COLOR_CURSOR_STATS_BG);
    painter.line_segment(
        [rect.left_top(), rect.left_bottom()],
        Stroke::new(1.0, COLOR_CURSOR_STATS_BORDER),
    );
    painter.line_segment(
        [rect.right_top(), rect.right_bottom()],
        Stroke::new(1.0, COLOR_CURSOR_STATS_BORDER),
    );

    let delete_rect = Rect::from_min_size(rect.min + vec2(2.0, 3.0), vec2(18.0, rect.height() - 6.0));
    let delete_response = ui.interact(
        delete_rect,
        ui.make_persistent_id("delete-secondary-cursor"),
        Sense::click(),
    );
    painter.rect_stroke(
        delete_rect.shrink(1.0),
        2.0,
        Stroke::new(
            1.0,
            if delete_response.hovered() {
                COLOR_CURSOR
            } else {
                COLOR_CURSOR_DELETE_BORDER
            },
        ),
    );
    painter.text(
        delete_rect.center(),
        Align2::CENTER_CENTER,
        "×",
        FontId::proportional(13.0),
        COLOR_CURSOR,
    );

    let delta = cursor_delta(cursor_time, secondary_cursor_time);
    painter
        .with_clip_rect(Rect::from_min_max(
            pos2(delete_rect.right() + 3.0, rect.top()),
            rect.right_bottom(),
        ))
        .text(
            pos2(delete_rect.right() + 4.0, rect.center().y),
            Align2::LEFT_CENTER,
            format!("t2:{secondary_cursor_time} Δ:{delta:+}"),
            FontId::monospace(10.5),
            COLOR_CURSOR,
        );

    delete_response.clicked()
}

pub fn cursor_delta(cursor_time: u64, secondary_cursor_time: u64) -> i128 {
    secondary_cursor_time as i128 - cursor_time as i128
}

pub fn edge_counts(changes: &[WaveChange], bit_offset: usize, bit_len: usize, start: u64, end: u64) -> EdgeCounts {
    let (start, end) = if start <= end { (start, end) } else { (end, start) };
    let mut previous = changes
        .iter()
        .rev()
        .find(|change| change.time < start)
        .map(|change| change.bits.as_slice());
    let mut counts = EdgeCounts::default();
    for change in changes
        .iter()
        .filter(|change| change.time >= start && change.time < end)
    {
        if let Some(previous_bits) = previous {
            if !bits_equal(previous_bits, &change.bits, bit_offset, bit_len) {
                counts.toggles += 1;
                if bit_len == 1 {
                    let before = get_bit(previous_bits, bit_offset);
                    let after = get_bit(&change.bits, bit_offset);
                    match (before, after) {
                        (false, true) => counts.posedges += 1,
                        (true, false) => counts.negedges += 1,
                        _ => {}
                    }
                }
            }
        }
        previous = Some(change.bits.as_slice());
    }
    counts
}

fn bits_equal(a: &[u8], b: &[u8], bit_offset: usize, bit_len: usize) -> bool {
    (0..bit_len).all(|index| get_bit(a, bit_offset + index) == get_bit(b, bit_offset + index))
}

fn get_bit(bits: &[u8], bit: usize) -> bool {
    bits.get(bit / 8).is_some_and(|byte| ((byte >> (bit % 8)) & 1) != 0)
}

fn major_tick_step(pixels_per_time: f32) -> u64 {
    (80.0 / pixels_per_time).ceil().max(1.0) as u64
}

fn first_tick_at_or_after(time: f32, tick_step: u64) -> u64 {
    ((time / tick_step as f32).ceil() as u64).saturating_mul(tick_step)
}

pub fn draw_time_grid(painter: &egui::Painter, rect: Rect, pixels_per_time: f32, time_view_start: f32, max_time: u64) {
    let tick_step = major_tick_step(pixels_per_time);
    let mut t = first_tick_at_or_after(time_view_start, tick_step);
    while t <= max_time {
        let x = time_to_x(rect, t as f32, time_view_start, pixels_per_time);
        if x > rect.right() {
            break;
        }
        painter.line_segment(
            [pos2(x, rect.top()), pos2(x, rect.bottom())],
            Stroke::new(1.0, COLOR_TIME_GRID),
        );
        t = t.saturating_add(tick_step);
    }
}

pub fn time_to_x(rect: Rect, time: f32, time_view_start: f32, pixels_per_time: f32) -> f32 {
    rect.left() + (time - time_view_start) * pixels_per_time
}

pub fn draw_cursor(painter: &egui::Painter, rect: Rect, cursor_time: u64, pixels_per_time: f32, time_view_start: f32) {
    let x = time_to_x(rect, cursor_time as f32, time_view_start, pixels_per_time);
    if x >= rect.left() && x <= rect.right() {
        painter.line_segment(
            [pos2(x, rect.top()), pos2(x, rect.bottom())],
            Stroke::new(1.5, COLOR_CURSOR),
        );
    }
}

pub fn draw_dotted_cursor(
    painter: &egui::Painter,
    rect: Rect,
    cursor_time: u64,
    pixels_per_time: f32,
    time_view_start: f32,
) {
    let x = time_to_x(rect, cursor_time as f32, time_view_start, pixels_per_time);
    if x < rect.left() || x > rect.right() {
        return;
    }
    let mut y = rect.top();
    while y < rect.bottom() {
        let y_end = (y + 7.0).min(rect.bottom());
        painter.line_segment([pos2(x, y), pos2(x, y_end)], Stroke::new(2.5, COLOR_CURSOR));
        y += 12.0;
    }
}

pub fn draw_zoom_selection(
    painter: &egui::Painter,
    rect: Rect,
    start_time: u64,
    current_time: u64,
    pixels_per_time: f32,
    time_view_start: f32,
) {
    let x0 = time_to_x(rect, start_time as f32, time_view_start, pixels_per_time);
    let x1 = time_to_x(rect, current_time as f32, time_view_start, pixels_per_time);
    let left = x0.min(x1).clamp(rect.left(), rect.right());
    let right = x0.max(x1).clamp(rect.left(), rect.right());
    painter.rect_filled(
        Rect::from_min_max(rect.left_top(), pos2(left, rect.bottom())),
        0.0,
        COLOR_ZOOM_SHADE,
    );
    painter.rect_filled(
        Rect::from_min_max(pos2(right, rect.top()), rect.right_bottom()),
        0.0,
        COLOR_ZOOM_SHADE,
    );
    painter.line_segment(
        [pos2(left, rect.top()), pos2(left, rect.bottom())],
        Stroke::new(2.0, COLOR_TEXT_STRONG),
    );
    painter.line_segment(
        [pos2(right, rect.top()), pos2(right, rect.bottom())],
        Stroke::new(2.0, COLOR_TEXT_STRONG),
    );
}

#[cfg(test)]
mod tests {
    use super::{cursor_delta, edge_counts, zoom_to_selection};
    use hwl_language::sim::recorder::WaveChange;

    #[test]
    fn edge_counts_between_cursors_count_directional_edges() {
        let changes = vec![
            WaveChange { time: 0, bits: vec![0] },
            WaveChange { time: 1, bits: vec![1] },
            WaveChange { time: 2, bits: vec![0] },
            WaveChange { time: 3, bits: vec![1] },
        ];

        let counts = edge_counts(&changes, 0, 1, 0, 3);

        assert_eq!(counts.posedges, 1);
        assert_eq!(counts.negedges, 1);
        assert_eq!(counts.toggles, 2);
    }

    #[test]
    fn edge_counts_include_start_and_exclude_end() {
        let changes = vec![
            WaveChange { time: 0, bits: vec![0] },
            WaveChange { time: 1, bits: vec![1] },
            WaveChange { time: 2, bits: vec![0] },
        ];

        let counts = edge_counts(&changes, 0, 1, 1, 2);
        assert_eq!(counts.posedges, 1);
        assert_eq!(counts.negedges, 0);
        assert_eq!(counts.toggles, 1);

        let reversed = edge_counts(&changes, 0, 1, 2, 1);
        assert_eq!(reversed.posedges, counts.posedges);
        assert_eq!(reversed.negedges, counts.negedges);
        assert_eq!(reversed.toggles, counts.toggles);
    }

    #[test]
    fn zoom_to_selection_fits_range_when_not_clamped() {
        let (time_view_start, pixels_per_time) = zoom_to_selection(900.0, 100, 13, 4).unwrap();

        assert_eq!(time_view_start, 4.0);
        assert_eq!(pixels_per_time, 100.0);
    }

    #[test]
    fn zoom_to_selection_uses_release_order_and_clamps_to_trace_end() {
        let (time_view_start, pixels_per_time) = zoom_to_selection(900.0, 20, 18, 10).unwrap();

        assert_eq!(pixels_per_time, 112.5);
        assert_eq!(time_view_start, 10.0);
        assert!(zoom_to_selection(900.0, 20, 10, 10).is_none());
    }

    #[test]
    fn cursor_delta_is_signed_second_minus_primary() {
        assert_eq!(cursor_delta(10, 15), 5);
        assert_eq!(cursor_delta(15, 10), -5);
    }
}
