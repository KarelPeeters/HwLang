use eframe::egui::{
    self, Align2, CentralPanel, Color32, Context, FontId, Rect, ScrollArea, Sense, SidePanel, Stroke, TopBottomPanel,
    Ui, pos2, vec2,
};
use hwl_language::sim::recorder::{WaveSignal, WaveSignalType, WaveStore};
use std::collections::BTreeSet;
use std::path::PathBuf;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions::default();
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
        }
    }
}

impl eframe::App for WaveGuiApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        TopBottomPanel::top("toolbar").show(ctx, |ui| self.toolbar(ui));
        SidePanel::left("hierarchy").resizable(true).show(ctx, |ui| {
            ui.heading("Hierarchy");
            if let Some(store) = &self.store {
                draw_hierarchy(ui, store, &mut self.rows, &[]);
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
            ui.text_edit_singleline(&mut self.path);
            if ui.button("Load").clicked() {
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
            ui.label(&self.status);
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
        let wave_width = max_time as f32 * self.pixels_per_time + 120.0;

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_width(280.0);
                self.row_controls(ui, &store);
            });
            ui.separator();
            ScrollArea::horizontal().show(ui, |ui| {
                ui.set_min_width(wave_width);
                draw_time_axis(ui, max_time, self.pixels_per_time, wave_width);
                let mut remove = None;
                let mut move_up = None;
                let mut move_down = None;
                for (row_index, signal_id) in self.rows.iter().copied().enumerate() {
                    if let Some(signal) = store.signals.get(signal_id) {
                        draw_signal_rows(
                            ui,
                            &store,
                            signal,
                            self.cursor_time,
                            self.pixels_per_time,
                            &mut RowAction {
                                row_index,
                                rows_len: self.rows.len(),
                                remove: &mut remove,
                                move_up: &mut move_up,
                                move_down: &mut move_down,
                            },
                        );
                    }
                }
                if let Some(row) = remove {
                    self.rows.remove(row);
                }
                if let Some(row) = move_up {
                    self.rows.swap(row, row - 1);
                }
                if let Some(row) = move_down {
                    self.rows.swap(row, row + 1);
                }
            });
        });
    }

    fn row_controls(&mut self, ui: &mut Ui, store: &WaveStore) {
        ui.heading("Rows");
        if self.rows.is_empty() {
            ui.label("Add signals from the hierarchy.");
        }
        for signal_id in &self.rows {
            if let Some(signal) = store.signals.get(*signal_id) {
                ui.label(format!("{}.{}", signal.path.join("."), signal.name));
            }
        }
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

struct RowAction<'a> {
    row_index: usize,
    rows_len: usize,
    remove: &'a mut Option<usize>,
    move_up: &'a mut Option<usize>,
    move_down: &'a mut Option<usize>,
}

fn draw_hierarchy(ui: &mut Ui, store: &WaveStore, rows: &mut Vec<usize>, prefix: &[String]) {
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
            .show(ui, |ui| draw_hierarchy(ui, store, rows, &next_prefix));
    }

    for signal in store.signals.iter().filter(|signal| signal.path == prefix) {
        ui.horizontal(|ui| {
            if ui.button("+").clicked() && !rows.contains(&signal.id) {
                rows.push(signal.id);
            }
            ui.label(&signal.name);
        });
    }
}

fn draw_time_axis(ui: &mut Ui, max_time: u64, pixels_per_time: f32, width: f32) {
    let height = 24.0;
    let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.line_segment(
        [pos2(rect.left(), rect.bottom()), pos2(rect.right(), rect.bottom())],
        Stroke::new(1.0, Color32::GRAY),
    );

    let tick_step = (80.0 / pixels_per_time).ceil().max(1.0) as u64;
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
}

fn draw_signal_rows(
    ui: &mut Ui,
    store: &WaveStore,
    signal: &WaveSignal,
    cursor_time: u64,
    pixels_per_time: f32,
    action: &mut RowAction<'_>,
) {
    match &signal.ty {
        WaveSignalType::Array { .. } | WaveSignalType::Tuple(_) | WaveSignalType::Struct { .. } => {
            egui::CollapsingHeader::new(signal.name.clone())
                .default_open(false)
                .show(ui, |ui| {
                    draw_signal_leaf(ui, store, signal, 0, signal.bit_len, cursor_time, pixels_per_time, action);
                    for (name, ty, offset, len) in composite_children(&signal.ty) {
                        draw_child_leaf(
                            ui,
                            store,
                            signal,
                            &name,
                            offset,
                            len.max(ty.bit_len()),
                            cursor_time,
                            pixels_per_time,
                        );
                    }
                });
        }
        _ => draw_signal_leaf(ui, store, signal, 0, signal.bit_len, cursor_time, pixels_per_time, action),
    }
}

fn draw_signal_leaf(
    ui: &mut Ui,
    store: &WaveStore,
    signal: &WaveSignal,
    bit_offset: usize,
    bit_len: usize,
    cursor_time: u64,
    pixels_per_time: f32,
    action: &mut RowAction<'_>,
) {
    ui.horizontal(|ui| {
        if ui.small_button("x").clicked() {
            *action.remove = Some(action.row_index);
        }
        if action.row_index > 0 && ui.small_button("up").clicked() {
            *action.move_up = Some(action.row_index);
        }
        if action.row_index + 1 < action.rows_len && ui.small_button("down").clicked() {
            *action.move_down = Some(action.row_index);
        }
        let value = store
            .signal_value_at(signal.id, cursor_time)
            .map(|bits| format_value(bits, bit_offset, bit_len))
            .unwrap_or_else(|| "x".to_owned());
        ui.label(format!("{} = {value}", signal.name));
    });
    draw_waveform(ui, &store.changes[signal.id], bit_offset, bit_len, pixels_per_time);
}

fn draw_child_leaf(
    ui: &mut Ui,
    store: &WaveStore,
    signal: &WaveSignal,
    name: &str,
    bit_offset: usize,
    bit_len: usize,
    cursor_time: u64,
    pixels_per_time: f32,
) {
    ui.horizontal(|ui| {
        let value = store
            .signal_value_at(signal.id, cursor_time)
            .map(|bits| format_value(bits, bit_offset, bit_len))
            .unwrap_or_else(|| "x".to_owned());
        ui.label(format!("  {name} = {value}"));
    });
    draw_waveform(ui, &store.changes[signal.id], bit_offset, bit_len, pixels_per_time);
}

fn draw_waveform(
    ui: &mut Ui,
    changes: &[hwl_language::sim::recorder::WaveChange],
    bit_offset: usize,
    bit_len: usize,
    pixels_per_time: f32,
) {
    let max_time = changes.last().map_or(1, |change| change.time.max(1));
    let width = max_time as f32 * pixels_per_time + 120.0;
    let height = 28.0;
    let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(rect, 0.0, Stroke::new(1.0, Color32::DARK_GRAY));

    if changes.is_empty() {
        return;
    }

    if bit_len == 1 {
        let mut prev_x = rect.left();
        let mut prev_y = bit_y(rect, get_bit(&changes[0].bits, bit_offset));
        for change in changes.iter().skip(1) {
            let x = rect.left() + change.time as f32 * pixels_per_time;
            let y = bit_y(rect, get_bit(&change.bits, bit_offset));
            painter.line_segment([pos2(prev_x, prev_y), pos2(x, prev_y)], Stroke::new(1.5, Color32::LIGHT_GREEN));
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
            draw_bus_segment(
                &painter,
                rect,
                last.time,
                max_time + 1,
                &format_value(&last.bits, bit_offset, bit_len),
                pixels_per_time,
            );
        }
    }
}

fn bit_y(rect: Rect, value: bool) -> f32 {
    if value {
        rect.top() + 6.0
    } else {
        rect.bottom() - 6.0
    }
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
    painter.line_segment([pos2(x0, y0), pos2(x0 + 4.0, y1)], Stroke::new(1.0, Color32::LIGHT_BLUE));
    painter.line_segment([pos2(x0, y1), pos2(x0 + 4.0, y0)], Stroke::new(1.0, Color32::LIGHT_BLUE));
    painter.text(
        pos2(x0 + 6.0, rect.center().y),
        Align2::LEFT_CENTER,
        label,
        FontId::monospace(11.0),
        Color32::WHITE,
    );
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
