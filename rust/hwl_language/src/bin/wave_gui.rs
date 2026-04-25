use eframe::egui::{
    self, Align2, CentralPanel, Color32, Context, FontId, Key, Rect, ScrollArea, Sense, SidePanel, Stroke,
    TopBottomPanel, Ui, ViewportBuilder, pos2, vec2,
};
use hwl_language::sim::recorder::{WaveSignal, WaveSignalType, WaveStore};
use std::collections::BTreeSet;
use std::path::PathBuf;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: ViewportBuilder::default()
            .with_title("HWL Wave GUI")
            .with_inner_size([1280.0, 900.0]),
        ..Default::default()
    };
    eframe::run_native(
        "HWL Wave GUI",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::<WaveGuiApp>::default())
        }),
    )
}

struct WaveGuiApp {
    store: Option<WaveStore>,
    rows: Vec<usize>,
    path: String,
    status: String,
    pixels_per_time: f32,
    cursor_time: u64,
    run_to_time: u64,
    step_count: u64,
    selected_row: Option<usize>,
    selected_signal: Option<usize>,
    last_hierarchy_click: Option<(usize, f64)>,
    dragging_row: Option<usize>,
    dragging_signal: Option<usize>,
    expanded_rows: BTreeSet<usize>,
}

impl Default for WaveGuiApp {
    fn default() -> Self {
        Self {
            store: None,
            rows: Vec::new(),
            path: String::new(),
            status: String::new(),
            pixels_per_time: 10.0,
            cursor_time: 0,
            run_to_time: 0,
            step_count: 1,
            selected_row: None,
            selected_signal: None,
            last_hierarchy_click: None,
            dragging_row: None,
            dragging_signal: None,
            expanded_rows: BTreeSet::new(),
        }
    }
}

impl eframe::App for WaveGuiApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        TopBottomPanel::top("toolbar").show(ctx, |ui| self.toolbar(ui));
        SidePanel::left("hierarchy")
            .resizable(true)
            .default_width(260.0)
            .width_range(180.0..=420.0)
            .show(ctx, |ui| {
                ui.heading("Hierarchy");
                if let Some(store) = self.store.clone() {
                    ui.horizontal(|ui| {
                        if ui.button("Clear").clicked() {
                            self.rows.clear();
                            self.selected_row = None;
                            self.selected_signal = None;
                            self.last_hierarchy_click = None;
                            self.dragging_row = None;
                            self.dragging_signal = None;
                            self.expanded_rows.clear();
                        }
                    });
                    ui.separator();
                    ui.label("Double-click or drag signals into the waveform pane.");
                    if ui.input(|input| input.key_pressed(Key::Enter)) {
                        if let Some(signal_id) = self.selected_signal {
                            toggle_row(&mut self.rows, signal_id);
                        }
                    }
                    ScrollArea::vertical().show(ui, |ui| {
                        draw_hierarchy(
                            ui,
                            &store,
                            &mut self.rows,
                            &mut self.selected_signal,
                            &mut self.last_hierarchy_click,
                            &mut self.dragging_signal,
                            &[],
                        );
                    });
                } else {
                    ui.label("Load a WaveStore JSON file to browse signals.");
                }
            });
        CentralPanel::default().show(ctx, |ui| self.wave_panel(ui));
    }
}

impl WaveGuiApp {
    fn toolbar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Store:");
            let path_response = ui.add_sized([360.0, 20.0], egui::TextEdit::singleline(&mut self.path));
            let load_requested =
                ui.button("Load").clicked() || path_response.lost_focus() && ui.input(|input| input.key_pressed(Key::Enter));
            if load_requested {
                self.load_store();
            }
            if ui.button("Save").clicked() {
                self.save_store();
            }
            ui.separator();
            ui.label("Zoom:");
            ui.add(egui::Slider::new(&mut self.pixels_per_time, 1.0..=80.0).logarithmic(true));
            ui.separator();
            ui.label("Time:");
            ui.add(egui::DragValue::new(&mut self.cursor_time).speed(1));
            ui.label("Step N:");
            ui.add(egui::DragValue::new(&mut self.step_count).speed(1));
            if ui.button("Step").clicked() {
                self.cursor_time = self.cursor_time.saturating_add(self.step_count.max(1));
            }
            ui.label("Run to:");
            ui.add(egui::DragValue::new(&mut self.run_to_time).speed(1));
            if ui.button("Run").clicked() {
                self.cursor_time = self.run_to_time;
            }
        });
        if !self.status.is_empty() {
            ui.colored_label(Color32::LIGHT_BLUE, &self.status);
        }
    }

    fn wave_panel(&mut self, ui: &mut Ui) {
        let Some(store) = self.store.clone() else {
            ui.centered_and_justified(|ui| {
                ui.label("No waveform store loaded.");
            });
            return;
        };

        let max_time = store.max_time().max(self.cursor_time).max(1);
        let visible_wave_width = (ui.available_width() - ROW_LABEL_WIDTH).max(200.0);
        let wave_width = (max_time as f32 * self.pixels_per_time + 120.0).max(visible_wave_width);

        if ui.input(|input| input.key_pressed(Key::Delete)) {
            if let Some(row_index) = self.selected_row.take() {
                if row_index < self.rows.len() {
                    let removed = self.rows.remove(row_index);
                    self.expanded_rows.remove(&removed);
                    if row_index < self.rows.len() {
                        self.selected_row = Some(row_index);
                    } else if !self.rows.is_empty() {
                        self.selected_row = Some(self.rows.len() - 1);
                    }
                }
            }
        }
        if !ui.input(|input| input.pointer.primary_down()) {
            self.dragging_row = None;
        }
        if let Some(signal_id) = self.dragging_signal {
            if ui.input(|input| input.pointer.any_released()) {
                if ui.rect_contains_pointer(ui.max_rect()) && !self.rows.contains(&signal_id) {
                    self.rows.push(signal_id);
                    self.selected_row = Some(self.rows.len() - 1);
                }
                self.dragging_signal = None;
            }
        }

        if self.rows.is_empty() {
            ui.label("Double-click a hierarchy signal or drag it here to add it.");
            return;
        }

        ui.horizontal_wrapped(|ui| {
            ui.weak("Click a waveform to move the cursor. Double-click composite rows to expand/collapse. Drag rows to reorder. Select a row and press Delete to remove.");
        });

        ScrollArea::both().show(ui, |ui| {
            ui.set_min_width(ROW_LABEL_WIDTH + wave_width);
            draw_time_axis(ui, max_time, self.cursor_time, self.pixels_per_time, wave_width);
            ui.separator();
            let mut reorder: Option<(usize, usize)> = None;
            let pointer_pos = ui.input(|input| input.pointer.interact_pos());
            for (row_index, signal_id) in self.rows.iter().copied().enumerate() {
                if let Some(signal) = store.signals.get(signal_id) {
                    let result = draw_signal_rows(
                        ui,
                        &store,
                        signal,
                        row_index,
                        self.selected_row == Some(row_index),
                        self.dragging_row == Some(row_index),
                        self.expanded_rows.contains(&signal.id),
                        self.cursor_time,
                        self.pixels_per_time,
                        wave_width,
                    );
                    if result.clicked {
                        self.selected_row = Some(row_index);
                    }
                    if let Some(time) = result.clicked_wave_time {
                        self.cursor_time = time;
                    }
                    if result.double_clicked && is_composite(&signal.ty) {
                        if !self.expanded_rows.remove(&signal.id) {
                            self.expanded_rows.insert(signal.id);
                        }
                    }
                    if result.drag_started {
                        self.selected_row = Some(row_index);
                        self.dragging_row = Some(row_index);
                    }
                    if let (Some(from), Some(pointer_pos)) = (self.dragging_row, pointer_pos) {
                        if from != row_index
                            && result.rect.contains(pointer_pos)
                            && ui.input(|input| input.pointer.primary_down())
                        {
                            reorder = Some((from, row_index));
                        }
                    }
                }
            }
            if let Some((from, to)) = reorder {
                let moved = self.rows.remove(from);
                self.rows.insert(to, moved);
                self.selected_row = Some(to);
                self.dragging_row = Some(to);
            }
        });
    }

    fn load_store(&mut self) {
        let path = PathBuf::from(self.path.trim());
        match std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<WaveStore>(&s).map_err(|e| e.to_string()))
        {
            Ok(store) => {
                self.cursor_time = store.max_time();
                self.rows.clear();
                self.selected_row = None;
                self.selected_signal = None;
                self.last_hierarchy_click = None;
                self.dragging_row = None;
                self.dragging_signal = None;
                self.expanded_rows.clear();
                self.store = Some(store);
                self.status = format!("Loaded {}", path.display());
            }
            Err(err) => {
                self.status = format!("Load failed: {err}");
            }
        }
    }

    fn save_store(&mut self) {
        let Some(store) = &self.store else {
            self.status = "No store to save".to_owned();
            return;
        };
        let path = PathBuf::from(self.path.trim());
        match serde_json::to_string_pretty(store)
            .map_err(|e| e.to_string())
            .and_then(|s| std::fs::write(&path, s).map_err(|e| e.to_string()))
        {
            Ok(()) => {
                self.status = format!("Saved {}", path.display());
            }
            Err(err) => {
                self.status = format!("Save failed: {err}");
            }
        }
    }
}

struct RowResult {
    clicked: bool,
    double_clicked: bool,
    drag_started: bool,
    clicked_wave_time: Option<u64>,
    rect: Rect,
}

const ROW_LABEL_WIDTH: f32 = 360.0;
const ROW_HEIGHT: f32 = 28.0;

fn draw_hierarchy(
    ui: &mut Ui,
    store: &WaveStore,
    rows: &mut Vec<usize>,
    selected_signal: &mut Option<usize>,
    last_hierarchy_click: &mut Option<(usize, f64)>,
    dragging_signal: &mut Option<usize>,
    prefix: &[String],
) {
    let mut child_paths = BTreeSet::new();
    for signal in &store.signals {
        if signal.path.starts_with(prefix) {
            if let Some(child) = signal.path.get(prefix.len()) {
                child_paths.insert(child.clone());
            }
        }
    }

    for child in child_paths {
        let mut next_prefix = prefix.to_owned();
        next_prefix.push(child.clone());
        egui::CollapsingHeader::new(child)
            .default_open(prefix.is_empty())
            .show(ui, |ui| {
                draw_hierarchy(
                    ui,
                    store,
                    rows,
                    selected_signal,
                    last_hierarchy_click,
                    dragging_signal,
                    &next_prefix,
                )
            });
    }

    for signal in store.signals.iter().filter(|signal| signal.path == prefix) {
        let already_added = rows.contains(&signal.id);
        let selected = *selected_signal == Some(signal.id);
        let row_height = 20.0;
        let (rect, response) =
            ui.allocate_exact_size(vec2(ui.available_width(), row_height), Sense::click_and_drag());
        if selected {
            ui.painter()
                .rect_filled(rect, 2.0, Color32::from_rgb(35, 55, 85));
        } else if already_added {
            ui.painter()
                .rect_filled(rect, 2.0, Color32::from_rgb(20, 80, 110));
        } else if response.hovered() {
            ui.painter()
                .rect_filled(rect, 2.0, Color32::from_rgb(35, 35, 35));
        }
        ui.painter().text(
            pos2(rect.left() + 4.0, rect.center().y),
            Align2::LEFT_CENTER,
            &signal.name,
            FontId::proportional(13.0),
            Color32::LIGHT_GRAY,
        );
        if response.clicked() {
            *selected_signal = Some(signal.id);
            let now = ui.input(|input| input.time);
            let repeated_click = last_hierarchy_click
                .is_some_and(|(last_id, last_time)| last_id == signal.id && now - last_time <= 0.45);
            if response.double_clicked() || repeated_click {
                toggle_row(rows, signal.id);
                *last_hierarchy_click = None;
            } else {
                *last_hierarchy_click = Some((signal.id, now));
            }
        }
        if response.drag_started() || response.is_pointer_button_down_on() {
            *selected_signal = Some(signal.id);
            *dragging_signal = Some(signal.id);
        }
    }
}

fn toggle_row(rows: &mut Vec<usize>, signal_id: usize) {
    if rows.contains(&signal_id) {
        rows.retain(|&row| row != signal_id);
    } else {
        rows.push(signal_id);
    }
}

fn draw_time_axis(ui: &mut Ui, max_time: u64, cursor_time: u64, pixels_per_time: f32, width: f32) {
    let height = 24.0;
    let (row_rect, _) = ui.allocate_exact_size(vec2(ROW_LABEL_WIDTH + width, height), Sense::hover());
    let rect = Rect::from_min_size(pos2(row_rect.left() + ROW_LABEL_WIDTH, row_rect.top()), vec2(width, height));
    let painter = ui.painter_at(rect);
    painter.line_segment(
        [pos2(rect.left(), rect.bottom()), pos2(rect.right(), rect.bottom())],
        Stroke::new(1.0, Color32::GRAY),
    );
    draw_time_grid(&painter, rect, pixels_per_time);

    let tick_step = major_tick_step(pixels_per_time);
    let mut t = 0;
    while t <= max_time {
        let x = rect.left() + t as f32 * pixels_per_time;
        painter.line_segment(
            [pos2(x, rect.bottom()), pos2(x, rect.bottom() - 6.0)],
            Stroke::new(1.0, Color32::GRAY),
        );
        painter.text(
            pos2(x + 2.0, rect.top()),
            Align2::LEFT_TOP,
            t.to_string(),
            FontId::monospace(11.0),
            Color32::GRAY,
        );
        t = t.saturating_add(tick_step);
    }
    draw_cursor(&painter, rect, cursor_time, pixels_per_time);
}

fn draw_signal_rows(
    ui: &mut Ui,
    store: &WaveStore,
    signal: &WaveSignal,
    row_index: usize,
    selected: bool,
    dragging: bool,
    expanded: bool,
    cursor_time: u64,
    pixels_per_time: f32,
    wave_width: f32,
) -> RowResult {
    let mut parent_result = draw_signal_leaf(
        ui,
        store,
        signal,
        &format!("{}.{}", signal.path.join("."), signal.name),
        row_index,
        selected,
        dragging,
        is_composite(&signal.ty),
        expanded,
        0,
        signal.bit_len,
        cursor_time,
        pixels_per_time,
        wave_width,
    );

    if expanded {
        for (name, ty, offset, len) in composite_children(&signal.ty) {
            let child_result = draw_signal_leaf(
                ui,
                store,
                signal,
                &format!("  {name}"),
                row_index,
                false,
                false,
                is_composite(ty),
                false,
                offset,
                len.max(ty.bit_len()),
                cursor_time,
                pixels_per_time,
                wave_width,
            );
            parent_result.clicked |= child_result.clicked;
            if parent_result.clicked_wave_time.is_none() {
                parent_result.clicked_wave_time = child_result.clicked_wave_time;
            }
        }
    }

    parent_result
}

fn draw_signal_leaf(
    ui: &mut Ui,
    store: &WaveStore,
    signal: &WaveSignal,
    label: &str,
    row_index: usize,
    selected: bool,
    dragging: bool,
    expandable: bool,
    expanded: bool,
    bit_offset: usize,
    bit_len: usize,
    cursor_time: u64,
    pixels_per_time: f32,
    wave_width: f32,
) -> RowResult {
    let (row_rect, response) =
        ui.allocate_exact_size(vec2(ROW_LABEL_WIDTH + wave_width, ROW_HEIGHT), Sense::click_and_drag());
    let label_rect = Rect::from_min_size(row_rect.min, vec2(ROW_LABEL_WIDTH, ROW_HEIGHT));
    let wave_rect = Rect::from_min_size(
        pos2(row_rect.left() + ROW_LABEL_WIDTH, row_rect.top()),
        vec2(wave_width, ROW_HEIGHT),
    );
    let painter = ui.painter_at(row_rect);

    let bg = if dragging {
        Color32::from_rgb(60, 50, 20)
    } else if selected {
        Color32::from_rgb(35, 55, 85)
    } else if response.hovered() {
        Color32::from_rgb(35, 35, 35)
    } else if row_index % 2 == 0 {
        Color32::from_rgb(20, 20, 20)
    } else {
        Color32::from_rgb(26, 26, 26)
    };
    painter.rect_filled(row_rect, 0.0, bg);

    let value = store
        .signal_value_at(signal.id, cursor_time)
        .map(|bits| format_value(bits, bit_offset, bit_len))
        .unwrap_or_else(|| "x".to_owned());
    let marker = if expandable {
        if expanded { "v " } else { "> " }
    } else {
        "  "
    };
    painter.with_clip_rect(label_rect).text(
        pos2(label_rect.left() + 6.0, label_rect.center().y),
        Align2::LEFT_CENTER,
        format!("{marker}{label} = {value}"),
        FontId::monospace(12.0),
        Color32::WHITE,
    );
    draw_waveform(
        &painter,
        wave_rect,
        &store.changes[signal.id],
        bit_offset,
        bit_len,
        cursor_time,
        pixels_per_time,
    );
    let clicked_wave_time = response
        .interact_pointer_pos()
        .filter(|pos| response.clicked() && wave_rect.contains(*pos))
        .map(|pos| ((pos.x - wave_rect.left()) / pixels_per_time).round().max(0.0) as u64);

    RowResult {
        clicked: response.clicked(),
        double_clicked: response.double_clicked(),
        drag_started: response.drag_started(),
        clicked_wave_time,
        rect: row_rect,
    }
}

fn draw_waveform(
    painter: &egui::Painter,
    rect: Rect,
    changes: &[hwl_language::sim::recorder::WaveChange],
    bit_offset: usize,
    bit_len: usize,
    cursor_time: u64,
    pixels_per_time: f32,
) {
    painter.rect_stroke(rect, 0.0, Stroke::new(1.0, Color32::DARK_GRAY));
    draw_time_grid(painter, rect, pixels_per_time);
    draw_cursor(painter, rect, cursor_time, pixels_per_time);

    if changes.is_empty() {
        return;
    }

    if bit_len == 1 {
        let mut prev_x = rect.left();
        let mut prev_y = bit_y(rect, get_bit(&changes[0].bits, bit_offset));
        for change in changes.iter().skip(1) {
            let x = rect.left() + change.time as f32 * pixels_per_time;
            let y = bit_y(rect, get_bit(&change.bits, bit_offset));
            painter.line_segment(
                [pos2(prev_x, prev_y), pos2(x, prev_y)],
                Stroke::new(1.5, Color32::LIGHT_GREEN),
            );
            painter.line_segment([pos2(x, prev_y), pos2(x, y)], Stroke::new(1.5, Color32::LIGHT_GREEN));
            prev_x = x;
            prev_y = y;
        }
        painter.line_segment(
            [pos2(prev_x, prev_y), pos2(rect.right(), prev_y)],
            Stroke::new(1.5, Color32::LIGHT_GREEN),
        );
    } else {
        for window in changes.windows(2) {
            let start = &window[0];
            let end = &window[1];
            draw_bus_segment(
                &painter,
                rect,
                start.time,
                end.time,
                &format_value(&start.bits, bit_offset, bit_len),
                pixels_per_time,
            );
        }
        if let Some(last) = changes.last() {
            let visible_end_time = (rect.width() / pixels_per_time).ceil().max(last.time as f32 + 1.0) as u64;
            draw_bus_segment(
                &painter,
                rect,
                last.time,
                visible_end_time,
                &format_value(&last.bits, bit_offset, bit_len),
                pixels_per_time,
            );
        }
    }
}

fn major_tick_step(pixels_per_time: f32) -> u64 {
    (80.0 / pixels_per_time).ceil().max(1.0) as u64
}

fn draw_time_grid(painter: &egui::Painter, rect: Rect, pixels_per_time: f32) {
    let tick_step = major_tick_step(pixels_per_time);
    let max_time = ((rect.width() / pixels_per_time).ceil() as u64).max(1);
    let mut t = 0;
    while t <= max_time {
        let x = rect.left() + t as f32 * pixels_per_time;
        painter.line_segment(
            [pos2(x, rect.top()), pos2(x, rect.bottom())],
            Stroke::new(1.0, Color32::from_gray(42)),
        );
        t = t.saturating_add(tick_step);
    }
}

fn draw_cursor(painter: &egui::Painter, rect: Rect, cursor_time: u64, pixels_per_time: f32) {
    let x = rect.left() + cursor_time as f32 * pixels_per_time;
    if x >= rect.left() && x <= rect.right() {
        painter.line_segment(
            [pos2(x, rect.top()), pos2(x, rect.bottom())],
            Stroke::new(1.5, Color32::from_rgb(240, 180, 80)),
        );
    }
}

fn is_composite(ty: &WaveSignalType) -> bool {
    matches!(
        ty,
        WaveSignalType::Array { .. } | WaveSignalType::Tuple(_) | WaveSignalType::Struct { .. }
    )
}

fn bit_y(rect: Rect, value: bool) -> f32 {
    if value { rect.top() + 6.0 } else { rect.bottom() - 6.0 }
}

fn draw_bus_segment(
    painter: &egui::Painter,
    rect: Rect,
    start_time: u64,
    end_time: u64,
    label: &str,
    pixels_per_time: f32,
) {
    let x0 = rect.left() + start_time as f32 * pixels_per_time;
    let x1 = rect.left() + end_time as f32 * pixels_per_time;
    let y0 = rect.top() + 5.0;
    let y1 = rect.bottom() - 5.0;
    painter.line_segment([pos2(x0, y0), pos2(x1, y0)], Stroke::new(1.0, Color32::LIGHT_BLUE));
    painter.line_segment([pos2(x0, y1), pos2(x1, y1)], Stroke::new(1.0, Color32::LIGHT_BLUE));
    painter.line_segment(
        [pos2(x0, y0), pos2(x0 + 4.0, y1)],
        Stroke::new(1.0, Color32::LIGHT_BLUE),
    );
    painter.line_segment(
        [pos2(x0, y1), pos2(x0 + 4.0, y0)],
        Stroke::new(1.0, Color32::LIGHT_BLUE),
    );

    let segment_width = x1 - x0;
    let estimated_text_width = label.len() as f32 * 7.0;
    if segment_width > estimated_text_width + 12.0 {
        let clip_rect = Rect::from_min_max(pos2(x0 + 4.0, rect.top()), pos2(x1 - 4.0, rect.bottom()));
        painter.with_clip_rect(clip_rect).text(
            pos2((x0 + x1) / 2.0, rect.center().y),
            Align2::CENTER_CENTER,
            label,
            FontId::monospace(11.0),
            Color32::WHITE,
        );
    }
}

fn composite_children(ty: &WaveSignalType) -> Vec<(String, &WaveSignalType, usize, usize)> {
    let mut result = Vec::new();
    match ty {
        WaveSignalType::Array { len, element } => {
            let stride = element.bit_len();
            for i in 0..*len {
                result.push((format!("[{i}]"), element.as_ref(), i * stride, stride));
            }
        }
        WaveSignalType::Tuple(elements) => {
            let mut offset = 0;
            for (i, element) in elements.iter().enumerate() {
                let len = element.bit_len();
                result.push((format!(".{i}"), element, offset, len));
                offset += len;
            }
        }
        WaveSignalType::Struct { fields, .. } => {
            let mut offset = 0;
            for (name, element) in fields {
                let len = element.bit_len();
                result.push((format!(".{name}"), element, offset, len));
                offset += len;
            }
        }
        _ => {}
    }
    result
}

fn format_value(bits: &[u8], bit_offset: usize, bit_len: usize) -> String {
    if bit_len == 0 {
        return "0b".to_owned();
    }
    if bit_len == 1 {
        return if get_bit(bits, bit_offset) { "1" } else { "0" }.to_owned();
    }
    let mut value = 0u128;
    let shown_bits = bit_len.min(128);
    for i in 0..shown_bits {
        if get_bit(bits, bit_offset + i) {
            value |= 1u128 << i;
        }
    }
    if bit_len <= 128 {
        format!("0x{value:x}")
    } else {
        format!("0x{value:x}...")
    }
}

fn get_bit(bits: &[u8], bit: usize) -> bool {
    bits.get(bit / 8).is_some_and(|byte| ((byte >> (bit % 8)) & 1) != 0)
}
