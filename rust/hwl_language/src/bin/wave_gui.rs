use eframe::egui::{
    self, Align2, CentralPanel, Color32, Context, FontId, Key, Rect, ScrollArea, Sense, Shape, SidePanel, Stroke,
    TopBottomPanel, Ui, ViewportBuilder, pos2, vec2,
};
use hwl_language::sim::recorder::{WaveSignal, WaveSignalKind, WaveSignalType, WaveStore};
use std::collections::BTreeSet;
use std::hash::Hash;
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
    selected_signals: BTreeSet<usize>,
    last_selected_signal: Option<usize>,
    last_hierarchy_click: Option<(usize, f64)>,
    row_drag: Option<RowDrag>,
    dragging_signals: Vec<usize>,
    expanded_rows: BTreeSet<WaveRowKey>,
}

#[derive(Debug, Clone)]
struct RowDrag {
    signal_id: usize,
    insert_index: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct WaveRowKey {
    signal_id: usize,
    bit_offset: usize,
    bit_len: usize,
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
            selected_signals: BTreeSet::new(),
            last_selected_signal: None,
            last_hierarchy_click: None,
            row_drag: None,
            dragging_signals: Vec::new(),
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
                            self.selected_signals.clear();
                            self.last_selected_signal = None;
                            self.last_hierarchy_click = None;
                            self.row_drag = None;
                            self.dragging_signals.clear();
                            self.expanded_rows.clear();
                        }
                    });
                    ui.separator();
                    ui.label("Click, Shift-click, or drag signals. Double-click modules/groups to add them.");
                    if ui.input(|input| input.key_pressed(Key::Enter)) {
                        if !self.selected_signals.is_empty() {
                            add_rows(&mut self.rows, self.selected_signals.iter().copied());
                        }
                    }
                    ScrollArea::vertical().show(ui, |ui| {
                        draw_hierarchy(
                            ui,
                            &store,
                            &mut self.rows,
                            &mut self.selected_signals,
                            &mut self.last_selected_signal,
                            &mut self.last_hierarchy_click,
                            &mut self.dragging_signals,
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

        let zoom_delta = ui.input(|input| {
            if input.modifiers.ctrl {
                input.raw_scroll_delta.y
            } else {
                0.0
            }
        });
        if zoom_delta != 0.0 {
            let factor = (1.0_f32 + zoom_delta.abs() / 240.0).clamp(1.02, 2.0);
            if zoom_delta > 0.0 {
                self.pixels_per_time = (self.pixels_per_time * factor).min(80.0);
            } else {
                self.pixels_per_time = (self.pixels_per_time / factor).max(1.0);
            }
        }

        let max_time = store.max_time().max(self.cursor_time).max(1);
        let visible_wave_width = (ui.available_width() - ROW_LABEL_WIDTH).max(200.0);
        let wave_width = (max_time as f32 * self.pixels_per_time + 120.0).max(visible_wave_width);

        if ui.input(|input| input.key_pressed(Key::Delete)) {
            if let Some(row_index) = self.selected_row.take() {
                if row_index < self.rows.len() {
                    let removed = self.rows.remove(row_index);
                    self.expanded_rows.retain(|key| key.signal_id != removed);
                    if row_index < self.rows.len() {
                        self.selected_row = Some(row_index);
                    } else if !self.rows.is_empty() {
                        self.selected_row = Some(self.rows.len() - 1);
                    }
                }
            }
        }
        if ui.input(|input| input.pointer.any_released()) {
            if let Some(row_drag) = self.row_drag.take() {
                let insert_index = row_drag.insert_index.min(self.rows.len());
                self.rows.insert(insert_index, row_drag.signal_id);
                self.selected_row = Some(insert_index);
            } else if !self.dragging_signals.is_empty() {
                if ui.rect_contains_pointer(ui.max_rect()) {
                    let start = self.rows.len();
                    add_rows(&mut self.rows, self.dragging_signals.iter().copied());
                    if self.rows.len() > start {
                        self.selected_row = Some(start);
                    }
                }
                self.dragging_signals.clear();
            }
        } else if !ui.input(|input| input.pointer.primary_down()) && self.row_drag.is_none() {
            self.dragging_signals.clear();
        }

        if self.rows.is_empty() && self.row_drag.is_none() {
            ui.label("Double-click a hierarchy signal or drag it here to add it.");
            return;
        }

        ui.horizontal_wrapped(|ui| {
            ui.weak("Drag labels vertically to reorder. Drag waveforms horizontally to move the cursor. Ctrl+scroll zooms horizontally. Select a row and press Delete to remove.");
        });

        ScrollArea::both().show(ui, |ui| {
            ui.set_min_width(ROW_LABEL_WIDTH + wave_width);
            let axis_rect = draw_time_axis(ui, max_time, self.pixels_per_time, wave_width);
            ui.separator();
            let pointer_pos = ui.input(|input| input.pointer.interact_pos());
            let mut start_row_drag: Option<usize> = None;
            let mut row_rects = Vec::new();
            let mut wave_bottom = axis_rect.bottom();
            for (row_index, signal_id) in self.rows.iter().copied().enumerate() {
                if let Some(signal) = store.signals.get(signal_id) {
                    let result = draw_signal_rows(
                        ui,
                        &store,
                        signal,
                        row_index,
                        self.selected_row == Some(row_index),
                        self.row_drag.as_ref().is_some_and(|drag| drag.signal_id == signal_id),
                        &self.expanded_rows,
                        self.cursor_time,
                        self.pixels_per_time,
                        wave_width,
                    );
                    if result.primary.clicked {
                        self.selected_row = Some(row_index);
                    }
                    if let Some(time) = result.cursor_time {
                        self.cursor_time = time;
                    }
                    for key in result.expand_toggles {
                        if !self.expanded_rows.remove(&key) {
                            self.expanded_rows.insert(key);
                        }
                    }
                    if result.primary.label_drag_started {
                        self.selected_row = Some(row_index);
                        start_row_drag = Some(row_index);
                    }
                    row_rects.push(result.primary.rect);
                    wave_bottom = wave_bottom.max(result.all_rect.bottom());
                }
            }

            if let (Some(row_drag), Some(pointer_pos)) = (&mut self.row_drag, pointer_pos) {
                row_drag.insert_index = insertion_index(pointer_pos, &row_rects);
            }
            if let Some(start_index) = start_row_drag {
                if start_index < self.rows.len() {
                    let signal_id = self.rows.remove(start_index);
                    let insert_index = pointer_pos.map_or(start_index, |pos| insertion_index(pos, &row_rects));
                    self.row_drag = Some(RowDrag { signal_id, insert_index });
                    self.selected_row = None;
                }
            }

            if let Some(row_drag) = &self.row_drag {
                draw_insert_line(ui, row_drag.insert_index, &row_rects, ROW_LABEL_WIDTH + wave_width);
            }
            let cursor_bottom = wave_bottom.max(ui.clip_rect().bottom());
            let cursor_span = Rect::from_min_max(axis_rect.min, pos2(axis_rect.right(), cursor_bottom));
            draw_cursor(ui.painter(), cursor_span, self.cursor_time, self.pixels_per_time);
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
                self.selected_signals.clear();
                self.last_selected_signal = None;
                self.last_hierarchy_click = None;
                self.row_drag = None;
                self.dragging_signals.clear();
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

#[derive(Clone)]
struct RowResult {
    clicked: bool,
    label_drag_started: bool,
    cursor_time: Option<u64>,
    expand_toggles: Vec<WaveRowKey>,
    rect: Rect,
}

struct SignalRowsResult {
    primary: RowResult,
    all_rect: Rect,
    cursor_time: Option<u64>,
    expand_toggles: Vec<WaveRowKey>,
}

const ROW_LABEL_WIDTH: f32 = 360.0;
const ROW_HEIGHT: f32 = 28.0;

fn draw_hierarchy(
    ui: &mut Ui,
    store: &WaveStore,
    rows: &mut Vec<usize>,
    selected_signals: &mut BTreeSet<usize>,
    last_selected_signal: &mut Option<usize>,
    last_hierarchy_click: &mut Option<(usize, f64)>,
    dragging_signals: &mut Vec<usize>,
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
        let subtree_signals = signal_ids_under_path(store, &next_prefix);
        let response = egui::CollapsingHeader::new(format!("{child} ({})", subtree_signals.len()))
            .default_open(prefix.is_empty())
            .show(ui, |ui| {
                draw_hierarchy(
                    ui,
                    store,
                    rows,
                    selected_signals,
                    last_selected_signal,
                    last_hierarchy_click,
                    dragging_signals,
                    &next_prefix,
                )
            });
        if response.header_response.clicked() {
            update_group_selection(ui, &subtree_signals, selected_signals, last_selected_signal);
        }
        if response.header_response.double_clicked() {
            add_rows(rows, subtree_signals.iter().copied());
        }
        if response.header_response.drag_started() || response.header_response.is_pointer_button_down_on() {
            *dragging_signals = subtree_signals;
        }
    }

    let ports = signal_ids_in_path(store, prefix, WaveSignalKind::Port);
    let wires = signal_ids_in_path(store, prefix, WaveSignalKind::Wire);
    draw_signal_group(
        ui,
        store,
        "[ports]",
        &ports,
        rows,
        selected_signals,
        last_selected_signal,
        last_hierarchy_click,
        dragging_signals,
    );
    draw_signal_group(
        ui,
        store,
        "[wires]",
        &wires,
        rows,
        selected_signals,
        last_selected_signal,
        last_hierarchy_click,
        dragging_signals,
    );
}

fn draw_signal_group(
    ui: &mut Ui,
    store: &WaveStore,
    title: &str,
    signal_ids: &[usize],
    rows: &mut Vec<usize>,
    selected_signals: &mut BTreeSet<usize>,
    last_selected_signal: &mut Option<usize>,
    last_hierarchy_click: &mut Option<(usize, f64)>,
    dragging_signals: &mut Vec<usize>,
) {
    if signal_ids.is_empty() {
        return;
    }

    let response = egui::CollapsingHeader::new(format!("{title} ({})", signal_ids.len()))
        .default_open(title == "[ports]")
        .show(ui, |ui| {
            for signal_id in signal_ids {
                let signal = &store.signals[*signal_id];
                draw_hierarchy_signal(
                    ui,
                    store,
                    signal,
                    rows,
                    selected_signals,
                    last_selected_signal,
                    last_hierarchy_click,
                    dragging_signals,
                );
            }
        });
    if response.header_response.clicked() {
        update_group_selection(ui, signal_ids, selected_signals, last_selected_signal);
    }
    if response.header_response.double_clicked() {
        add_rows(rows, signal_ids.iter().copied());
    }
    if response.header_response.drag_started() || response.header_response.is_pointer_button_down_on() {
        *dragging_signals = signal_ids.to_vec();
    }
}

fn draw_hierarchy_signal(
    ui: &mut Ui,
    store: &WaveStore,
    signal: &WaveSignal,
    rows: &mut Vec<usize>,
    selected_signals: &mut BTreeSet<usize>,
    last_selected_signal: &mut Option<usize>,
    last_hierarchy_click: &mut Option<(usize, f64)>,
    dragging_signals: &mut Vec<usize>,
) {
        let already_added = rows.contains(&signal.id);
        let selected = selected_signals.contains(&signal.id);
        let row_height = 20.0;
        let (rect, response) =
            ui.allocate_exact_size(vec2(ui.available_width(), row_height), Sense::click_and_drag());
        if selected {
            ui.painter()
                .rect_filled(rect, 2.0, Color32::from_rgb(35, 55, 85));
        } else if already_added {
            ui.painter()
                .rect_filled(rect, 2.0, Color32::from_rgb(55, 85, 35));
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
            update_signal_selection(ui, store, signal.id, selected_signals, last_selected_signal);
            let now = ui.input(|input| input.time);
            let repeated_click = last_hierarchy_click
                .is_some_and(|(last_id, last_time)| last_id == signal.id && now - last_time <= 0.45);
            if response.double_clicked() || repeated_click {
                if selected_signals.len() > 1 && selected_signals.contains(&signal.id) {
                    add_rows(rows, selected_signals.iter().copied());
                } else {
                    toggle_row(rows, signal.id);
                }
                *last_hierarchy_click = None;
            } else {
                *last_hierarchy_click = Some((signal.id, now));
            }
        }
        if response.drag_started() || response.is_pointer_button_down_on() {
            if !selected_signals.contains(&signal.id) {
                selected_signals.clear();
                selected_signals.insert(signal.id);
                *last_selected_signal = Some(signal.id);
            }
            *dragging_signals = selected_signals.iter().copied().collect();
        }
}

fn toggle_row(rows: &mut Vec<usize>, signal_id: usize) {
    if rows.contains(&signal_id) {
        rows.retain(|&row| row != signal_id);
    } else {
        rows.push(signal_id);
    }
}

fn add_rows(rows: &mut Vec<usize>, signal_ids: impl IntoIterator<Item = usize>) {
    for signal_id in signal_ids {
        if !rows.contains(&signal_id) {
            rows.push(signal_id);
        }
    }
}

fn signal_ids_under_path(store: &WaveStore, path: &[String]) -> Vec<usize> {
    store
        .signals
        .iter()
        .filter(|signal| signal.path.starts_with(path))
        .map(|signal| signal.id)
        .collect()
}

fn signal_ids_in_path(store: &WaveStore, path: &[String], kind: WaveSignalKind) -> Vec<usize> {
    store
        .signals
        .iter()
        .filter(|signal| signal.path == path && signal.kind == kind)
        .map(|signal| signal.id)
        .collect()
}

fn update_signal_selection(
    ui: &Ui,
    store: &WaveStore,
    signal_id: usize,
    selected_signals: &mut BTreeSet<usize>,
    last_selected_signal: &mut Option<usize>,
) {
    let modifiers = ui.input(|input| input.modifiers);
    if modifiers.shift {
        if let Some(anchor) = *last_selected_signal {
            let (start, end) = if anchor <= signal_id {
                (anchor, signal_id)
            } else {
                (signal_id, anchor)
            };
            for signal in &store.signals {
                if (start..=end).contains(&signal.id) {
                    selected_signals.insert(signal.id);
                }
            }
        } else {
            selected_signals.insert(signal_id);
        }
    } else if modifiers.ctrl || modifiers.command {
        if !selected_signals.remove(&signal_id) {
            selected_signals.insert(signal_id);
        }
        *last_selected_signal = Some(signal_id);
    } else {
        selected_signals.clear();
        selected_signals.insert(signal_id);
        *last_selected_signal = Some(signal_id);
    }
}

fn update_group_selection(
    ui: &Ui,
    signal_ids: &[usize],
    selected_signals: &mut BTreeSet<usize>,
    last_selected_signal: &mut Option<usize>,
) {
    if signal_ids.is_empty() {
        return;
    }
    let modifiers = ui.input(|input| input.modifiers);
    if !(modifiers.shift || modifiers.ctrl || modifiers.command) {
        selected_signals.clear();
    }
    if modifiers.ctrl || modifiers.command {
        let all_selected = signal_ids.iter().all(|signal_id| selected_signals.contains(signal_id));
        if all_selected {
            for signal_id in signal_ids {
                selected_signals.remove(signal_id);
            }
        } else {
            selected_signals.extend(signal_ids.iter().copied());
        }
    } else {
        selected_signals.extend(signal_ids.iter().copied());
    }
    *last_selected_signal = signal_ids.last().copied();
}

fn draw_time_axis(ui: &mut Ui, max_time: u64, pixels_per_time: f32, width: f32) -> Rect {
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
    rect
}

fn draw_signal_rows(
    ui: &mut Ui,
    store: &WaveStore,
    signal: &WaveSignal,
    row_index: usize,
    selected: bool,
    dragging: bool,
    expanded_rows: &BTreeSet<WaveRowKey>,
    cursor_time: u64,
    pixels_per_time: f32,
    wave_width: f32,
) -> SignalRowsResult {
    let key = WaveRowKey {
        signal_id: signal.id,
        bit_offset: 0,
        bit_len: signal.bit_len,
    };
    let result = draw_signal_tree(
        ui,
        store,
        signal,
        &format!("{}.{}", signal.path.join("."), signal.name),
        &signal.ty,
        key,
        row_index,
        selected,
        dragging,
        true,
        0,
        expanded_rows,
        cursor_time,
        pixels_per_time,
        wave_width,
    );
    result
}

fn draw_signal_tree(
    ui: &mut Ui,
    store: &WaveStore,
    signal: &WaveSignal,
    label: &str,
    ty: &WaveSignalType,
    key: WaveRowKey,
    row_index: usize,
    selected: bool,
    dragging: bool,
    can_reorder: bool,
    depth: usize,
    expanded_rows: &BTreeSet<WaveRowKey>,
    cursor_time: u64,
    pixels_per_time: f32,
    wave_width: f32,
) -> SignalRowsResult {
    let expanded = is_composite(ty) && expanded_rows.contains(&key);
    let leaf = draw_signal_leaf(
        ui,
        store,
        signal,
        label,
        row_index,
        selected && can_reorder,
        dragging && can_reorder,
        is_composite(ty),
        expanded,
        key,
        can_reorder,
        depth,
        cursor_time,
        pixels_per_time,
        wave_width,
    );
    let mut result = SignalRowsResult {
        primary: leaf.clone(),
        all_rect: leaf.rect,
        cursor_time: leaf.cursor_time,
        expand_toggles: leaf.expand_toggles,
    };

    if expanded {
        for (name, child_ty, offset, len) in composite_children(ty) {
            let child_key = WaveRowKey {
                signal_id: signal.id,
                bit_offset: key.bit_offset + offset,
                bit_len: len.max(child_ty.bit_len()),
            };
            let child = draw_signal_tree(
                ui,
                store,
                signal,
                &name,
                &child_ty,
                child_key,
                row_index,
                false,
                false,
                false,
                depth + 1,
                expanded_rows,
                cursor_time,
                pixels_per_time,
                wave_width,
            );
            result.all_rect = result.all_rect.union(child.all_rect);
            if result.cursor_time.is_none() {
                result.cursor_time = child.cursor_time;
            }
            result.expand_toggles.extend(child.expand_toggles);
        }
    }

    result
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
    key: WaveRowKey,
    can_reorder: bool,
    depth: usize,
    cursor_time: u64,
    pixels_per_time: f32,
    wave_width: f32,
) -> RowResult {
    let (row_rect, _) = ui.allocate_exact_size(vec2(ROW_LABEL_WIDTH + wave_width, ROW_HEIGHT), Sense::hover());
    let label_rect = Rect::from_min_size(row_rect.min, vec2(ROW_LABEL_WIDTH, ROW_HEIGHT));
    let wave_rect = Rect::from_min_size(
        pos2(row_rect.left() + ROW_LABEL_WIDTH, row_rect.top()),
        vec2(wave_width, ROW_HEIGHT),
    );
    let label_response = ui.interact(label_rect, ui.make_persistent_id(("row-label", key)), Sense::click_and_drag());
    let wave_response = ui.interact(wave_rect, ui.make_persistent_id(("row-wave", key)), Sense::click_and_drag());
    let painter = ui.painter_at(row_rect);

    let bg = if dragging {
        Color32::from_rgb(60, 50, 20)
    } else if selected {
        Color32::from_rgb(35, 55, 85)
    } else if label_response.hovered() || wave_response.hovered() {
        Color32::from_rgb(35, 35, 35)
    } else if row_index % 2 == 0 {
        Color32::from_rgb(20, 20, 20)
    } else {
        Color32::from_rgb(26, 26, 26)
    };
    painter.rect_filled(row_rect, 0.0, bg);

    let value = store
        .signal_value_at(signal.id, cursor_time)
        .map(|bits| format_value(bits, key.bit_offset, key.bit_len))
        .unwrap_or_else(|| "x".to_owned());
    let icon_rect = Rect::from_min_size(
        pos2(label_rect.left() + 4.0 + depth as f32 * 14.0, label_rect.center().y - 6.0),
        vec2(12.0, 12.0),
    );
    let icon_response = if expandable {
        Some(ui.interact(icon_rect, ui.make_persistent_id(("row-expand", key)), Sense::click()))
    } else {
        None
    };
    if expandable {
        draw_disclosure_icon(ui.painter(), icon_rect, expanded);
    }
    painter.with_clip_rect(label_rect).text(
        pos2(icon_rect.right() + 4.0, label_rect.center().y),
        Align2::LEFT_CENTER,
        format!("{label} = {value}"),
        FontId::monospace(12.0),
        Color32::WHITE,
    );
    draw_waveform(
        &painter,
        wave_rect,
        &store.changes[signal.id],
        key.bit_offset,
        key.bit_len,
        pixels_per_time,
    );
    let cursor_time = wave_response
        .interact_pointer_pos()
        .filter(|pos| (wave_response.clicked() || wave_response.dragged()) && wave_rect.contains(*pos))
        .map(|pos| ((pos.x - wave_rect.left()) / pixels_per_time).round().max(0.0) as u64);
    let expand_toggles = icon_response
        .filter(|response| response.clicked())
        .map(|_| vec![key])
        .unwrap_or_default();

    RowResult {
        clicked: label_response.clicked() || wave_response.clicked(),
        label_drag_started: can_reorder && label_response.drag_started(),
        cursor_time,
        expand_toggles,
        rect: row_rect,
    }
}

fn draw_waveform(
    painter: &egui::Painter,
    rect: Rect,
    changes: &[hwl_language::sim::recorder::WaveChange],
    bit_offset: usize,
    bit_len: usize,
    pixels_per_time: f32,
) {
    painter.rect_stroke(rect, 0.0, Stroke::new(1.0, Color32::DARK_GRAY));
    draw_time_grid(painter, rect, pixels_per_time);

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

fn enum_tag_width(variant_count: usize) -> usize {
    if variant_count <= 1 {
        0
    } else {
        usize::BITS as usize - (variant_count - 1).leading_zeros() as usize
    }
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

fn insertion_index(pointer_pos: egui::Pos2, row_rects: &[Rect]) -> usize {
    row_rects
        .iter()
        .position(|rect| pointer_pos.y < rect.center().y)
        .unwrap_or(row_rects.len())
}

fn draw_insert_line(ui: &Ui, insert_index: usize, row_rects: &[Rect], width: f32) {
    let Some(anchor) = row_rects
        .get(insert_index)
        .map(|rect| rect.top())
        .or_else(|| row_rects.last().map(|rect| rect.bottom()))
    else {
        return;
    };
    let left = row_rects
        .first()
        .map(|rect| rect.left())
        .unwrap_or_else(|| ui.min_rect().left());
    ui.painter().line_segment(
        [pos2(left, anchor), pos2(left + width, anchor)],
        Stroke::new(2.0, Color32::RED),
    );
}

fn draw_disclosure_icon(painter: &egui::Painter, rect: Rect, expanded: bool) {
    let points = if expanded {
        vec![
            pos2(rect.left() + 2.0, rect.top() + 4.0),
            pos2(rect.right() - 2.0, rect.top() + 4.0),
            pos2(rect.center().x, rect.bottom() - 2.0),
        ]
    } else {
        vec![
            pos2(rect.left() + 4.0, rect.top() + 2.0),
            pos2(rect.left() + 4.0, rect.bottom() - 2.0),
            pos2(rect.right() - 2.0, rect.center().y),
        ]
    };
    painter.add(Shape::convex_polygon(points, Color32::LIGHT_GRAY, Stroke::NONE));
}

fn is_composite(ty: &WaveSignalType) -> bool {
    matches!(
        ty,
        WaveSignalType::Array { .. }
            | WaveSignalType::Tuple(_)
            | WaveSignalType::Struct { .. }
            | WaveSignalType::Enum { .. }
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

fn composite_children(ty: &WaveSignalType) -> Vec<(String, WaveSignalType, usize, usize)> {
    let mut result = Vec::new();
    match ty {
        WaveSignalType::Array { len, element } => {
            let stride = element.bit_len();
            for i in 0..*len {
                result.push((format!("[{i}]"), element.as_ref().clone(), i * stride, stride));
            }
        }
        WaveSignalType::Tuple(elements) => {
            let mut offset = 0;
            for (i, element) in elements.iter().enumerate() {
                let len = element.bit_len();
                result.push((format!(".{i}"), element.clone(), offset, len));
                offset += len;
            }
        }
        WaveSignalType::Struct { fields, .. } => {
            let mut offset = 0;
            for (name, element) in fields {
                let len = element.bit_len();
                result.push((format!(".{name}"), element.clone(), offset, len));
                offset += len;
            }
        }
        WaveSignalType::Enum { variants, .. } => {
            let tag_width = enum_tag_width(variants.len());
            result.push((".tag".to_owned(), WaveSignalType::Int { signed: false, width: tag_width }, 0, tag_width));
            let payload_offset = tag_width;
            for (name, payload) in variants {
                if let Some(payload) = payload {
                    let len = payload.bit_len();
                    result.push((format!(".{name}"), payload.clone(), payload_offset, len));
                }
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
