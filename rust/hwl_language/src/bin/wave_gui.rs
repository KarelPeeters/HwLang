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
    rows: Vec<WaveRow>,
    next_row_id: u64,
    path: String,
    status: String,
    pixels_per_time: f32,
    row_label_width: f32,
    cursor_time: u64,
    run_to_time: u64,
    step_count: u64,
    selected_row: Option<u64>,
    selected_rows: BTreeSet<u64>,
    last_selected_row: Option<u64>,
    selected_modules: BTreeSet<Vec<String>>,
    last_selected_module: Option<Vec<String>>,
    selected_signals: BTreeSet<usize>,
    last_selected_signal: Option<usize>,
    row_drag: Option<RowDrag>,
    cursor_dragging: bool,
    dragging_signals: Vec<usize>,
    collapsed_hierarchy: BTreeSet<String>,
    expanded_rows: BTreeSet<WaveRowKey>,
    show_ports: bool,
    show_signals: bool,
}

#[derive(Debug, Clone)]
struct RowDrag {
    rows: Vec<WaveRow>,
    insert_index: usize,
}

#[derive(Debug, Clone)]
struct WaveRow {
    id: u64,
    kind: WaveRowKind,
}

#[derive(Debug, Clone)]
enum WaveRowKind {
    Signal { signal_id: usize, group: Option<u64> },
    Group { name: String, collapsed: bool },
}

impl WaveRow {
    fn signal_id(&self) -> Option<usize> {
        match self.kind {
            WaveRowKind::Signal { signal_id, .. } => Some(signal_id),
            WaveRowKind::Group { .. } => None,
        }
    }

    fn group_id(&self) -> Option<u64> {
        match self.kind {
            WaveRowKind::Signal { group, .. } => group,
            WaveRowKind::Group { .. } => None,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct WaveRowKey {
    row_id: u64,
    signal_id: usize,
    bit_offset: usize,
    bit_len: usize,
    part: u64,
}

impl Default for WaveGuiApp {
    fn default() -> Self {
        Self {
            store: None,
            rows: Vec::new(),
            next_row_id: 1,
            path: String::new(),
            status: String::new(),
            pixels_per_time: 10.0,
            row_label_width: DEFAULT_ROW_LABEL_WIDTH,
            cursor_time: 0,
            run_to_time: 0,
            step_count: 1,
            selected_row: None,
            selected_rows: BTreeSet::new(),
            last_selected_row: None,
            selected_modules: BTreeSet::new(),
            last_selected_module: None,
            selected_signals: BTreeSet::new(),
            last_selected_signal: None,
            row_drag: None,
            cursor_dragging: false,
            dragging_signals: Vec::new(),
            collapsed_hierarchy: BTreeSet::new(),
            expanded_rows: BTreeSet::new(),
            show_ports: true,
            show_signals: true,
        }
    }
}

impl eframe::App for WaveGuiApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        TopBottomPanel::top("toolbar").show(ctx, |ui| self.toolbar(ui));
        if !ctx.wants_keyboard_input()
            && ctx.input(|input| input.key_pressed(Key::W) && (input.modifiers.ctrl || input.modifiers.command))
        {
            if let Some(store) = self.store.clone() {
                self.add_selected_or_visible_signals(&store);
            }
        }
        SidePanel::left("hierarchy")
            .resizable(true)
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Hierarchy");
                if let Some(store) = self.store.clone() {
                    ui.horizontal(|ui| {
                        if ui.button("Clear").clicked() {
                            self.rows.clear();
                            self.next_row_id = 1;
                            self.selected_row = None;
                            self.selected_rows.clear();
                            self.last_selected_row = None;
                            self.selected_modules.clear();
                            self.last_selected_module = None;
                            self.selected_signals.clear();
                            self.last_selected_signal = None;
                            self.row_drag = None;
                            self.cursor_dragging = false;
                            self.dragging_signals.clear();
                            self.collapsed_hierarchy.clear();
                            self.expanded_rows.clear();
                        }
                    });
                    ScrollArea::vertical().show(ui, |ui| {
                        draw_hierarchy(
                            ui,
                            &store,
                            &mut self.selected_modules,
                            &mut self.last_selected_module,
                            &mut self.selected_signals,
                            &mut self.last_selected_signal,
                            &mut self.collapsed_hierarchy,
                            &[],
                        );
                    });
                } else {
                }
            });
        SidePanel::left("signals")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                ui.heading("Signals");
                if let Some(store) = self.store.clone() {
                    self.signal_panel(ui, &store);
                } else {
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
                ui.memory_mut(|memory| memory.surrender_focus(path_response.id));
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

        let max_time = store.max_time().max(1);
        let label_width = self.row_label_width;
        let visible_wave_width = (ui.available_width() - label_width).max(200.0);
        let wave_width = (max_time as f32 * self.pixels_per_time + 120.0).max(visible_wave_width);
        let visible_rows = visible_wave_rows(&self.rows);

        let text_editing = ui.ctx().wants_keyboard_input();
        if !text_editing && ui.input(|input| input.key_pressed(Key::A) && input.modifiers.ctrl) {
            self.selected_rows = visible_rows.iter().map(|row| row.row_id).collect();
            self.selected_row = visible_rows.first().map(|row| row.row_id);
            self.last_selected_row = self.selected_row;
        }
        if !text_editing
            && ui.input(|input| input.key_pressed(Key::G) && (input.modifiers.ctrl || input.modifiers.command))
        {
            self.create_group_from_selection();
        }
        if ui.input(|input| input.key_pressed(Key::Delete)) {
            if !self.selected_rows.is_empty() {
                delete_selected_rows(&mut self.rows, &self.selected_rows);
                self.expanded_rows
                    .retain(|key| self.rows.iter().any(|row| row.id == key.row_id));
                self.selected_rows.clear();
                self.selected_row = None;
                self.last_selected_row = None;
            }
        }
        if ui.input(|input| input.pointer.any_released()) {
            self.cursor_dragging = false;
            if let Some(row_drag) = self.row_drag.take() {
                let visible_rows = visible_wave_rows(&self.rows);
                let insert_index = actual_insert_index(&visible_rows, row_drag.insert_index, self.rows.len());
                let first_id = row_drag.rows.first().map(|row| row.id);
                self.rows.splice(insert_index..insert_index, row_drag.rows);
                self.selected_row = first_id;
                self.selected_rows.clear();
                if let Some(first_id) = first_id {
                    self.selected_rows.insert(first_id);
                }
                self.last_selected_row = first_id;
            }
        } else if !ui.input(|input| input.pointer.primary_down()) && self.row_drag.is_none() {
            self.dragging_signals.clear();
        }

        if self.rows.is_empty() && self.row_drag.is_none() && self.dragging_signals.is_empty() {
            return;
        }

        ScrollArea::both().show(ui, |ui| {
            ui.set_min_width(label_width + wave_width);
            let axis_rect = draw_time_axis(ui, self.pixels_per_time, label_width, wave_width);
            ui.separator();
            let pointer_pos = ui.input(|input| input.pointer.interact_pos());
            let mut start_row_drag: Option<usize> = None;
            let mut row_rects = Vec::new();
            let mut wave_bottom = axis_rect.bottom();
            let visible_rows = visible_wave_rows(&self.rows);
            for (visible_index, visible_row) in visible_rows.iter().enumerate() {
                let row_selected =
                    self.selected_rows.contains(&visible_row.row_id) || self.selected_row == Some(visible_row.row_id);
                match visible_row.kind {
                    VisibleWaveRowKind::Group => {
                        let result = draw_group_row(
                            ui,
                            &mut self.rows[visible_row.row_index],
                            visible_index,
                            row_selected,
                            label_width,
                            wave_width,
                        );
                        if result.clicked {
                            update_wave_row_selection(
                                ui,
                                visible_index,
                                &visible_rows,
                                &mut self.selected_rows,
                                &mut self.selected_row,
                                &mut self.last_selected_row,
                            );
                        }
                        if result.label_drag_started {
                            update_wave_row_selection(
                                ui,
                                visible_index,
                                &visible_rows,
                                &mut self.selected_rows,
                                &mut self.selected_row,
                                &mut self.last_selected_row,
                            );
                            start_row_drag = Some(visible_row.row_index);
                        }
                        row_rects.push(result.rect);
                        wave_bottom = wave_bottom.max(result.rect.bottom());
                    }
                    VisibleWaveRowKind::Signal { signal_id } => {
                        if let Some(signal) = store.signals.get(signal_id) {
                            let result = draw_signal_rows(
                                ui,
                                &store,
                                signal,
                                visible_row.row_id,
                                visible_index,
                                row_selected,
                                self.row_drag.as_ref().is_some_and(|drag| {
                                    drag.rows.iter().any(|row| row.id == visible_row.row_id)
                                }),
                                &self.expanded_rows,
                                self.cursor_time,
                                self.pixels_per_time,
                                label_width,
                                wave_width,
                            );
                            if result.primary.clicked {
                                update_wave_row_selection(
                                    ui,
                                    visible_index,
                                    &visible_rows,
                                    &mut self.selected_rows,
                                    &mut self.selected_row,
                                    &mut self.last_selected_row,
                                );
                            }
                            if let Some(time) = result.cursor_time {
                                self.cursor_time = time;
                            }
                            if result.cursor_drag_started {
                                self.cursor_dragging = true;
                            }
                            for key in result.expand_toggles {
                                if !self.expanded_rows.remove(&key) {
                                    self.expanded_rows.insert(key);
                                }
                            }
                            if result.primary.label_drag_started {
                                update_wave_row_selection(
                                    ui,
                                    visible_index,
                                    &visible_rows,
                                    &mut self.selected_rows,
                                    &mut self.selected_row,
                                    &mut self.last_selected_row,
                                );
                                start_row_drag = Some(visible_row.row_index);
                            }
                            row_rects.push(result.primary.rect);
                            wave_bottom = wave_bottom.max(result.all_rect.bottom());
                        }
                    }
                }
            }

            if let (Some(row_drag), Some(pointer_pos)) = (&mut self.row_drag, pointer_pos) {
                row_drag.insert_index = insertion_index(pointer_pos, &row_rects);
            }
            let left_drag_insert_index = if !self.dragging_signals.is_empty() {
                Some(pointer_pos.map_or(self.rows.len(), |pos| insertion_index(pos, &row_rects)))
            } else {
                None
            };
            if let Some(start_index) = start_row_drag {
                if start_index < self.rows.len() {
                    let rows = take_drag_rows(&mut self.rows, start_index);
                    let insert_index = pointer_pos.map_or(start_index, |pos| insertion_index(pos, &row_rects));
                    self.row_drag = Some(RowDrag { rows, insert_index });
                    self.selected_row = None;
                    self.selected_rows.clear();
                    self.last_selected_row = None;
                }
            }

            let fallback_insert_y = axis_rect.bottom() + 7.0;
            if let Some(row_drag) = &self.row_drag {
                draw_insert_line(
                    ui,
                    row_drag.insert_index,
                    &row_rects,
                    label_width + wave_width,
                    fallback_insert_y,
                );
            } else if let Some(insert_index) = left_drag_insert_index {
                draw_insert_line(ui, insert_index, &row_rects, label_width + wave_width, fallback_insert_y);
            }
            if ui.input(|input| input.pointer.any_released()) && !self.dragging_signals.is_empty() {
                if ui.rect_contains_pointer(ui.max_rect()) {
                    let visible_rows = visible_wave_rows(&self.rows);
                    let visible_insert_index = left_drag_insert_index.unwrap_or(visible_rows.len());
                    let insert_index = actual_insert_index(&visible_rows, visible_insert_index, self.rows.len());
                    let new_rows = self
                        .dragging_signals
                        .clone()
                        .into_iter()
                        .map(|signal_id| self.make_signal_row(signal_id, None))
                        .collect::<Vec<_>>();
                    let first_added = new_rows.first().map(|row| row.id);
                    self.rows.splice(insert_index..insert_index, new_rows);
                    if let Some(first_added) = first_added {
                        self.selected_rows.clear();
                        self.selected_rows.insert(first_added);
                        self.selected_row = Some(first_added);
                        self.last_selected_row = Some(first_added);
                    }
                }
                self.dragging_signals.clear();
            }
            let cursor_bottom = wave_bottom.max(ui.clip_rect().bottom());
            let cursor_span = Rect::from_min_max(axis_rect.min, pos2(axis_rect.right(), cursor_bottom));
            let splitter_rect = Rect::from_min_max(
                pos2(axis_rect.left() - 4.0, axis_rect.top()),
                pos2(axis_rect.left() + 4.0, cursor_bottom),
            );
            let splitter_response = ui.interact(splitter_rect, ui.make_persistent_id("wave-label-splitter"), Sense::drag());
            if splitter_response.dragged() {
                let delta_x = ui.input(|input| input.pointer.delta().x);
                self.row_label_width = (self.row_label_width + delta_x).max(MIN_ROW_LABEL_WIDTH);
            }
            ui.painter().line_segment(
                [pos2(axis_rect.left(), axis_rect.top()), pos2(axis_rect.left(), cursor_bottom)],
                Stroke::new(1.0, Color32::DARK_GRAY),
            );
            if self.cursor_dragging {
                if let Some(pointer_pos) = pointer_pos {
                    if ui.input(|input| input.pointer.primary_down()) {
                        self.cursor_time = time_from_pointer(axis_rect, pointer_pos, self.pixels_per_time);
                    }
                }
            }
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
                self.next_row_id = 1;
                self.selected_row = None;
                self.selected_rows.clear();
                self.last_selected_row = None;
                self.selected_modules.clear();
                if let Some(root_path) = store.signals.first().map(|signal| signal.path.clone()) {
                    self.selected_modules.insert(root_path.clone());
                    self.last_selected_module = Some(root_path);
                } else {
                    self.last_selected_module = None;
                }
                self.selected_signals.clear();
                self.last_selected_signal = None;
                self.row_drag = None;
                self.cursor_dragging = false;
                self.dragging_signals.clear();
                self.collapsed_hierarchy.clear();
                self.expanded_rows.clear();
                self.store = Some(store);
                self.status = format!("Loaded {}", path.display());
            }
            Err(err) => {
                self.status = format!("Load failed: {err}");
            }
        }
    }

    fn signal_panel(&mut self, ui: &mut Ui, store: &WaveStore) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_ports, "ports");
            ui.checkbox(&mut self.show_signals, "signals");
        });
        ui.separator();

        if self.selected_modules.is_empty() {
            return;
        }

        ui.horizontal(|ui| {
            ui.strong("Kind");
            ui.add_space(30.0);
            ui.strong("Signal");
        });
        ui.separator();

        let signal_ids = filtered_signal_ids(store, &self.selected_modules, self.show_ports, self.show_signals);
        self.selected_signals.retain(|signal_id| signal_ids.contains(signal_id));
        if signal_ids.is_empty() {
            return;
        }

        ScrollArea::vertical().show(ui, |ui| {
            for &signal_id in &signal_ids {
                draw_signal_panel_row(
                    ui,
                    store,
                    &signal_ids,
                    signal_id,
                    &self.rows,
                    &mut self.selected_signals,
                    &mut self.last_selected_signal,
                    &mut self.dragging_signals,
                );
            }
        });
    }

    fn add_selected_or_visible_signals(&mut self, store: &WaveStore) {
        let visible_signals = filtered_signal_ids(store, &self.selected_modules, self.show_ports, self.show_signals);
        self.selected_signals
            .retain(|signal_id| visible_signals.contains(signal_id));
        if self.selected_signals.is_empty() {
            let selected_modules = self.selected_modules.iter().cloned().collect::<Vec<_>>();
            for module_path in selected_modules {
                let module_signals = filtered_signal_ids_for_module(store, &module_path, self.show_ports, self.show_signals);
                if !module_signals.is_empty() {
                    let name = module_path.join(".");
                    self.add_grouped_signals(name, module_signals);
                }
            }
        } else {
            let signal_ids = self.selected_signals.iter().copied().collect::<Vec<_>>();
            self.add_signal_rows(signal_ids);
        }
    }

    fn next_row_id(&mut self) -> u64 {
        let id = self.next_row_id;
        self.next_row_id += 1;
        id
    }

    fn make_signal_row(&mut self, signal_id: usize, group: Option<u64>) -> WaveRow {
        WaveRow {
            id: self.next_row_id(),
            kind: WaveRowKind::Signal { signal_id, group },
        }
    }

    fn make_group_row(&mut self, name: String) -> WaveRow {
        WaveRow {
            id: self.next_row_id(),
            kind: WaveRowKind::Group {
                name,
                collapsed: false,
            },
        }
    }

    fn add_signal_rows(&mut self, signal_ids: impl IntoIterator<Item = usize>) {
        let rows = signal_ids
            .into_iter()
            .map(|signal_id| self.make_signal_row(signal_id, None))
            .collect::<Vec<_>>();
        self.rows.extend(rows);
    }

    fn add_grouped_signals(&mut self, name: String, signal_ids: impl IntoIterator<Item = usize>) {
        let group = self.make_group_row(name);
        let group_id = group.id;
        self.rows.push(group);
        let rows = signal_ids
            .into_iter()
            .map(|signal_id| self.make_signal_row(signal_id, Some(group_id)))
            .collect::<Vec<_>>();
        self.rows.extend(rows);
    }

    fn create_group_from_selection(&mut self) {
        if self.selected_rows.is_empty() {
            let group = self.make_group_row("group".to_owned());
            self.selected_rows.clear();
            self.selected_rows.insert(group.id);
            self.selected_row = Some(group.id);
            self.last_selected_row = Some(group.id);
            self.rows.push(group);
            return;
        }

        let selected = self.selected_rows.clone();
        let mut selected_indices = self
            .rows
            .iter()
            .enumerate()
            .filter_map(|(index, row)| match row.kind {
                WaveRowKind::Signal { .. } if selected.contains(&row.id) => Some(index),
                _ => None,
            })
            .collect::<Vec<_>>();
        if selected_indices.is_empty() {
            return;
        }
        selected_indices.sort_unstable();
        let original_insert_index = selected_indices[0];
        let group = self.make_group_row("group".to_owned());
        let group_id = group.id;

        let selected_set = selected_indices.into_iter().collect::<BTreeSet<_>>();
        let mut grouped_rows = Vec::new();
        let mut kept_rows = Vec::new();
        let mut insert_index = 0;
        for (index, mut row) in self.rows.drain(..).enumerate() {
            if selected_set.contains(&index) {
                if let WaveRowKind::Signal { group, .. } = &mut row.kind {
                    *group = Some(group_id);
                }
                grouped_rows.push(row);
            } else {
                if index < original_insert_index {
                    insert_index += 1;
                }
                kept_rows.push(row);
            }
        }
        self.rows = kept_rows;
        let insert_index = insert_index.min(self.rows.len());
        self.rows.splice(insert_index..insert_index, std::iter::once(group).chain(grouped_rows));
        self.selected_rows.clear();
        self.selected_rows.insert(group_id);
        self.selected_row = Some(group_id);
        self.last_selected_row = Some(group_id);
    }

}

#[derive(Clone)]
struct RowResult {
    clicked: bool,
    label_drag_started: bool,
    cursor_time: Option<u64>,
    cursor_drag_started: bool,
    expand_toggles: Vec<WaveRowKey>,
    rect: Rect,
}

struct SignalRowsResult {
    primary: RowResult,
    all_rect: Rect,
    cursor_time: Option<u64>,
    cursor_drag_started: bool,
    expand_toggles: Vec<WaveRowKey>,
}

struct TreeHeaderResult {
    clicked: bool,
    expanded: bool,
}

#[derive(Clone)]
struct VisibleWaveRow {
    row_index: usize,
    row_id: u64,
    kind: VisibleWaveRowKind,
}

#[derive(Clone)]
enum VisibleWaveRowKind {
    Signal { signal_id: usize },
    Group,
}

const DEFAULT_ROW_LABEL_WIDTH: f32 = 360.0;
const MIN_ROW_LABEL_WIDTH: f32 = 160.0;
const ROW_HEIGHT: f32 = 28.0;

fn draw_hierarchy(
    ui: &mut Ui,
    store: &WaveStore,
    selected_modules: &mut BTreeSet<Vec<String>>,
    last_selected_module: &mut Option<Vec<String>>,
    selected_signals: &mut BTreeSet<usize>,
    last_selected_signal: &mut Option<usize>,
    collapsed_hierarchy: &mut BTreeSet<String>,
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
        let key = hierarchy_key("module", &next_prefix);
        let has_submodules = store
            .signals
            .iter()
            .any(|signal| signal.path.starts_with(&next_prefix) && signal.path.len() > next_prefix.len());
        let header = draw_module_header(
            ui,
            &child,
            &key,
            &next_prefix,
            selected_modules,
            collapsed_hierarchy,
            has_submodules,
            0,
        );
        if header.clicked {
            update_module_selection(ui, store, &next_prefix, selected_modules, last_selected_module);
            selected_signals.clear();
            *last_selected_signal = None;
        }
        if header.expanded {
            ui.indent(key, |ui| {
                draw_hierarchy(
                    ui,
                    store,
                    selected_modules,
                    last_selected_module,
                    selected_signals,
                    last_selected_signal,
                    collapsed_hierarchy,
                    &next_prefix,
                )
            });
        }
    }
}

fn draw_signal_panel_row(
    ui: &mut Ui,
    store: &WaveStore,
    visible_signals: &[usize],
    signal_id: usize,
    rows: &[WaveRow],
    selected_signals: &mut BTreeSet<usize>,
    last_selected_signal: &mut Option<usize>,
    dragging_signals: &mut Vec<usize>,
) {
    let signal = &store.signals[signal_id];
    let already_added = rows.iter().any(|row| row.signal_id() == Some(signal.id));
    let selected = selected_signals.contains(&signal.id);
    let row_height = 22.0;
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), row_height), Sense::click_and_drag());
    if selected {
        ui.painter().rect_filled(rect, 2.0, Color32::from_rgb(35, 55, 85));
    } else if already_added {
        ui.painter().rect_filled(rect, 2.0, Color32::from_rgb(55, 85, 35));
    } else if response.hovered() {
        ui.painter().rect_filled(rect, 2.0, Color32::from_rgb(35, 35, 35));
    }
    let kind = match signal.kind {
        WaveSignalKind::Port => "port",
        WaveSignalKind::Wire => "signal",
    };
    ui.painter().text(
        pos2(rect.left() + 4.0, rect.center().y),
        Align2::LEFT_CENTER,
        kind,
        FontId::monospace(12.0),
        Color32::GRAY,
    );
    ui.painter().text(
        pos2(rect.left() + 64.0, rect.center().y),
        Align2::LEFT_CENTER,
        format!("{}.{}", signal.path.join("."), signal.name),
        FontId::proportional(13.0),
        Color32::LIGHT_GRAY,
    );
    if response.clicked() {
        update_signal_selection(ui, visible_signals, signal.id, selected_signals, last_selected_signal);
    }
    if response.drag_started() || response.dragged() {
        if !selected_signals.contains(&signal.id) {
            selected_signals.clear();
            selected_signals.insert(signal.id);
            *last_selected_signal = Some(signal.id);
        }
        *dragging_signals = selected_signals.iter().copied().collect();
    }
}

fn draw_module_header(
    ui: &mut Ui,
    title: &str,
    key: &str,
    path: &[String],
    selected_modules: &BTreeSet<Vec<String>>,
    collapsed_hierarchy: &mut BTreeSet<String>,
    has_children: bool,
    depth: usize,
) -> TreeHeaderResult {
    let height = 20.0;
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
    let indent = depth as f32 * 14.0;
    let icon_rect = Rect::from_min_size(pos2(rect.left() + 4.0 + indent, rect.center().y - 6.0), vec2(12.0, 12.0));
    let label_left = if has_children {
        icon_rect.right() + 4.0
    } else {
        rect.left() + 4.0 + indent
    };
    let label_rect = Rect::from_min_max(pos2(label_left, rect.top()), rect.right_bottom());
    let icon_response = if has_children {
        Some(ui.interact(icon_rect, ui.make_persistent_id(("module-icon", key)), Sense::click()))
    } else {
        None
    };
    let label_response = ui.interact(label_rect, ui.make_persistent_id(("module-label", key)), Sense::click());

    if icon_response.as_ref().is_some_and(|response| response.clicked()) {
        if !collapsed_hierarchy.remove(key) {
            collapsed_hierarchy.insert(key.to_owned());
        }
    }
    let expanded = has_children && !collapsed_hierarchy.contains(key);
    let bg = if selected_modules.contains(path) {
        Color32::from_rgb(35, 55, 85)
    } else if label_response.hovered() || icon_response.as_ref().is_some_and(|response| response.hovered()) {
        Color32::from_rgb(35, 35, 35)
    } else {
        Color32::TRANSPARENT
    };
    ui.painter().rect_filled(rect, 2.0, bg);
    if has_children {
        draw_disclosure_icon(ui.painter(), icon_rect, expanded);
    }
    ui.painter().text(
        pos2(label_rect.left(), label_rect.center().y),
        Align2::LEFT_CENTER,
        title,
        FontId::proportional(13.0),
        Color32::LIGHT_GRAY,
    );

    TreeHeaderResult {
        clicked: label_response.clicked(),
        expanded,
    }
}

fn hierarchy_key(kind: &str, path: &[String]) -> String {
    if path.is_empty() {
        kind.to_owned()
    } else {
        format!("{kind}:{}", path.join("."))
    }
}

fn filtered_signal_ids(
    store: &WaveStore,
    selected_modules: &BTreeSet<Vec<String>>,
    show_ports: bool,
    show_signals: bool,
) -> Vec<usize> {
    store
        .signals
        .iter()
        .filter(|signal| selected_modules.contains(&signal.path))
        .filter(|signal| match signal.kind {
            WaveSignalKind::Port => show_ports,
            WaveSignalKind::Wire => show_signals,
        })
        .map(|signal| signal.id)
        .collect()
}

fn filtered_signal_ids_for_module(
    store: &WaveStore,
    module_path: &[String],
    show_ports: bool,
    show_signals: bool,
) -> Vec<usize> {
    store
        .signals
        .iter()
        .filter(|signal| signal.path == module_path)
        .filter(|signal| match signal.kind {
            WaveSignalKind::Port => show_ports,
            WaveSignalKind::Wire => show_signals,
        })
        .map(|signal| signal.id)
        .collect()
}

fn module_paths(store: &WaveStore) -> Vec<Vec<String>> {
    store
        .signals
        .iter()
        .map(|signal| signal.path.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn visible_wave_rows(rows: &[WaveRow]) -> Vec<VisibleWaveRow> {
    let collapsed_groups = rows
        .iter()
        .filter_map(|row| match &row.kind {
            WaveRowKind::Group { collapsed: true, .. } => Some(row.id),
            _ => None,
        })
        .collect::<BTreeSet<_>>();

    rows.iter()
        .enumerate()
        .filter_map(|(row_index, row)| match row.kind {
            WaveRowKind::Group { .. } => Some(VisibleWaveRow {
                row_index,
                row_id: row.id,
                kind: VisibleWaveRowKind::Group,
            }),
            WaveRowKind::Signal { signal_id, group } => {
                if group.is_some_and(|group_id| collapsed_groups.contains(&group_id)) {
                    None
                } else {
                    Some(VisibleWaveRow {
                        row_index,
                        row_id: row.id,
                        kind: VisibleWaveRowKind::Signal { signal_id },
                    })
                }
            }
        })
        .collect()
}

fn actual_insert_index(visible_rows: &[VisibleWaveRow], visible_insert_index: usize, fallback: usize) -> usize {
    visible_rows
        .get(visible_insert_index)
        .map(|row| row.row_index)
        .unwrap_or(fallback)
}

fn take_drag_rows(rows: &mut Vec<WaveRow>, start_index: usize) -> Vec<WaveRow> {
    if start_index >= rows.len() {
        return Vec::new();
    }
    match rows[start_index].kind {
        WaveRowKind::Group { .. } => {
            let group_id = rows[start_index].id;
            let mut end = start_index + 1;
            while end < rows.len() && rows[end].group_id() == Some(group_id) {
                end += 1;
            }
            rows.drain(start_index..end).collect()
        }
        WaveRowKind::Signal { .. } => {
            let mut row = rows.remove(start_index);
            if let WaveRowKind::Signal { group, .. } = &mut row.kind {
                *group = None;
            }
            vec![row]
        }
    }
}

fn delete_selected_rows(rows: &mut Vec<WaveRow>, selected_rows: &BTreeSet<u64>) {
    let selected_groups = rows
        .iter()
        .filter_map(|row| match row.kind {
            WaveRowKind::Group { .. } if selected_rows.contains(&row.id) => Some(row.id),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    rows.retain(|row| {
        !selected_rows.contains(&row.id) && !row.group_id().is_some_and(|group_id| selected_groups.contains(&group_id))
    });
}

fn update_wave_row_selection(
    ui: &Ui,
    visible_index: usize,
    visible_rows: &[VisibleWaveRow],
    selected_rows: &mut BTreeSet<u64>,
    selected_row: &mut Option<u64>,
    last_selected_row: &mut Option<u64>,
) {
    let row_id = visible_rows[visible_index].row_id;
    let modifiers = ui.input(|input| input.modifiers);
    if modifiers.shift {
        if let Some(anchor) = *last_selected_row {
            let start = visible_rows.iter().position(|row| row.row_id == anchor);
            if let Some(start) = start {
                let (start, end) = if start <= visible_index {
                    (start, visible_index)
                } else {
                    (visible_index, start)
                };
                for row in &visible_rows[start..=end] {
                    selected_rows.insert(row.row_id);
                }
            } else {
                selected_rows.insert(row_id);
            }
        } else {
            selected_rows.insert(row_id);
        }
    } else if modifiers.ctrl || modifiers.command {
        if !selected_rows.remove(&row_id) {
            selected_rows.insert(row_id);
        }
        *last_selected_row = Some(row_id);
    } else {
        selected_rows.clear();
        selected_rows.insert(row_id);
        *last_selected_row = Some(row_id);
    }
    *selected_row = Some(row_id);
}

fn update_signal_selection(
    ui: &Ui,
    visible_signals: &[usize],
    signal_id: usize,
    selected_signals: &mut BTreeSet<usize>,
    last_selected_signal: &mut Option<usize>,
) {
    let modifiers = ui.input(|input| input.modifiers);
    if modifiers.shift {
        if let Some(anchor) = *last_selected_signal {
            let start = visible_signals.iter().position(|candidate| *candidate == anchor);
            let end = visible_signals.iter().position(|candidate| *candidate == signal_id);
            if let (Some(start), Some(end)) = (start, end) {
                let (start, end) = if start <= end { (start, end) } else { (end, start) };
                for visible_signal in &visible_signals[start..=end] {
                    selected_signals.insert(*visible_signal);
                }
            } else {
                selected_signals.insert(signal_id);
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

fn update_module_selection(
    ui: &Ui,
    store: &WaveStore,
    path: &[String],
    selected_modules: &mut BTreeSet<Vec<String>>,
    last_selected_module: &mut Option<Vec<String>>,
) {
    let modifiers = ui.input(|input| input.modifiers);
    let path = path.to_owned();
    if modifiers.shift {
        if let Some(anchor) = last_selected_module.clone() {
            let paths = module_paths(store);
            let start = paths.iter().position(|candidate| candidate == &anchor);
            let end = paths.iter().position(|candidate| candidate == &path);
            if let (Some(start), Some(end)) = (start, end) {
                let (start, end) = if start <= end { (start, end) } else { (end, start) };
                for module_path in &paths[start..=end] {
                    selected_modules.insert(module_path.clone());
                }
            } else {
                selected_modules.insert(path.clone());
            }
        } else {
            selected_modules.insert(path.clone());
        }
    } else if modifiers.ctrl || modifiers.command {
        if !selected_modules.remove(&path) {
            selected_modules.insert(path.clone());
        }
    } else {
        selected_modules.clear();
        selected_modules.insert(path.clone());
    }
    *last_selected_module = Some(path);
}

fn time_from_pointer(axis_rect: Rect, pointer_pos: egui::Pos2, pixels_per_time: f32) -> u64 {
    ((pointer_pos.x - axis_rect.left()) / pixels_per_time).round().max(0.0) as u64
}

fn draw_time_axis(ui: &mut Ui, pixels_per_time: f32, label_width: f32, width: f32) -> Rect {
    let height = 24.0;
    let (row_rect, _) = ui.allocate_exact_size(vec2(label_width + width, height), Sense::hover());
    let rect = Rect::from_min_size(pos2(row_rect.left() + label_width, row_rect.top()), vec2(width, height));
    let painter = ui.painter_at(rect);
    painter.line_segment(
        [pos2(rect.left(), rect.bottom()), pos2(rect.right(), rect.bottom())],
        Stroke::new(1.0, Color32::GRAY),
    );
    draw_time_grid(&painter, rect, pixels_per_time);

    let max_time = visible_time_span(rect, pixels_per_time);
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

fn draw_group_row(
    ui: &mut Ui,
    row: &mut WaveRow,
    row_index: usize,
    selected: bool,
    label_width: f32,
    wave_width: f32,
) -> RowResult {
    let (row_rect, _) = ui.allocate_exact_size(vec2(label_width + wave_width, ROW_HEIGHT), Sense::hover());
    let label_rect = Rect::from_min_size(row_rect.min, vec2(label_width, ROW_HEIGHT));
    let icon_hit_rect = Rect::from_min_size(pos2(label_rect.left(), label_rect.top()), vec2(28.0, ROW_HEIGHT));
    let icon_rect = Rect::from_center_size(icon_hit_rect.center(), vec2(12.0, 12.0));
    let drag_rect = Rect::from_min_max(label_rect.min, pos2(icon_hit_rect.right(), label_rect.bottom()));
    let drag_response = ui.interact(drag_rect, ui.make_persistent_id(("group-drag", row.id)), Sense::click_and_drag());
    let wave_rect = Rect::from_min_max(pos2(label_rect.right(), row_rect.top()), row_rect.right_bottom());
    let wave_response = ui.interact(wave_rect, ui.make_persistent_id(("group-wave", row.id)), Sense::click());
    let painter = ui.painter_at(row_rect);
    let bg = if selected {
        Color32::from_rgb(35, 55, 85)
    } else if drag_response.hovered() || wave_response.hovered() {
        Color32::from_rgb(35, 35, 35)
    } else if row_index % 2 == 0 {
        Color32::from_rgb(20, 20, 20)
    } else {
        Color32::from_rgb(26, 26, 26)
    };
    painter.rect_filled(row_rect, 0.0, bg);

    if let WaveRowKind::Group { name, collapsed } = &mut row.kind {
        let icon_response = ui.interact(icon_hit_rect, ui.make_persistent_id(("group-expand", row.id)), Sense::click());
        if icon_response.clicked() {
            *collapsed = !*collapsed;
        }
        draw_disclosure_icon(ui.painter(), icon_rect, !*collapsed);
        let edit_rect = Rect::from_min_max(pos2(icon_hit_rect.right(), label_rect.top() + 3.0), label_rect.right_bottom());
        ui.put(
            edit_rect,
            egui::TextEdit::singleline(name)
                .font(FontId::proportional(13.0))
                .frame(false),
        );
    }

    RowResult {
        clicked: drag_response.clicked() || wave_response.clicked(),
        label_drag_started: drag_response.drag_started(),
        cursor_time: None,
        cursor_drag_started: false,
        expand_toggles: Vec::new(),
        rect: row_rect,
    }
}

fn draw_signal_rows(
    ui: &mut Ui,
    store: &WaveStore,
    signal: &WaveSignal,
    row_id: u64,
    row_index: usize,
    selected: bool,
    dragging: bool,
    expanded_rows: &BTreeSet<WaveRowKey>,
    cursor_time: u64,
    pixels_per_time: f32,
    label_width: f32,
    wave_width: f32,
) -> SignalRowsResult {
    let key = WaveRowKey {
        row_id,
        signal_id: signal.id,
        bit_offset: 0,
        bit_len: signal.bit_len,
        part: 0,
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
        label_width,
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
    label_width: f32,
    wave_width: f32,
) -> SignalRowsResult {
    let expanded = is_composite(ty) && expanded_rows.contains(&key);
    let leaf = draw_signal_leaf(
        ui,
        store,
        signal,
        label,
        ty,
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
        label_width,
        wave_width,
    );
    let mut result = SignalRowsResult {
        primary: leaf.clone(),
        all_rect: leaf.rect,
        cursor_time: leaf.cursor_time,
        cursor_drag_started: leaf.cursor_drag_started,
        expand_toggles: leaf.expand_toggles,
    };

    if expanded {
        for (child_index, (name, child_ty, offset, len)) in composite_children(ty).into_iter().enumerate() {
            let child_key = WaveRowKey {
                signal_id: signal.id,
                row_id: key.row_id,
                bit_offset: key.bit_offset + offset,
                bit_len: len.max(child_ty.bit_len()),
                part: key.part.wrapping_mul(131).wrapping_add(child_index as u64 + 1),
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
                label_width,
                wave_width,
            );
            result.all_rect = result.all_rect.union(child.all_rect);
            if result.cursor_time.is_none() {
                result.cursor_time = child.cursor_time;
            }
            result.cursor_drag_started |= child.cursor_drag_started;
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
    ty: &WaveSignalType,
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
    label_width: f32,
    wave_width: f32,
) -> RowResult {
    let (row_rect, _) = ui.allocate_exact_size(vec2(label_width + wave_width, ROW_HEIGHT), Sense::hover());
    let label_rect = Rect::from_min_size(row_rect.min, vec2(label_width, ROW_HEIGHT));
    let wave_rect = Rect::from_min_size(
        pos2(row_rect.left() + label_width, row_rect.top()),
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
        .map(|bits| format_value_for_type(bits, key.bit_offset, ty))
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
        FontId::proportional(13.0),
        Color32::WHITE,
    );
    draw_waveform(
        &painter,
        wave_rect,
        &store.changes[signal.id],
        key.bit_offset,
        ty,
        pixels_per_time,
    );
    let cursor_time = wave_response
        .interact_pointer_pos()
        .filter(|pos| (wave_response.clicked() || wave_response.dragged()) && wave_rect.contains(*pos))
        .map(|pos| ((pos.x - wave_rect.left()) / pixels_per_time).round().max(0.0) as u64);
    let cursor_drag_started = wave_response.drag_started()
        || ui.input(|input| input.modifiers.shift) && wave_response.is_pointer_button_down_on();
    let expand_toggles = icon_response
        .filter(|response| response.clicked())
        .map(|_| vec![key])
        .unwrap_or_default();

    RowResult {
        clicked: label_response.clicked() || wave_response.clicked(),
        label_drag_started: can_reorder && label_response.drag_started(),
        cursor_time,
        cursor_drag_started,
        expand_toggles,
        rect: row_rect,
    }
}

fn draw_waveform(
    painter: &egui::Painter,
    rect: Rect,
    changes: &[hwl_language::sim::recorder::WaveChange],
    bit_offset: usize,
    ty: &WaveSignalType,
    pixels_per_time: f32,
) {
    painter.rect_stroke(rect, 0.0, Stroke::new(1.0, Color32::DARK_GRAY));
    draw_time_grid(painter, rect, pixels_per_time);

    if changes.is_empty() {
        return;
    }

    if ty.bit_len() == 1 {
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
                &format_value_for_type(&start.bits, bit_offset, ty),
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
                &format_value_for_type(&last.bits, bit_offset, ty),
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
    let max_time = visible_time_span(rect, pixels_per_time);
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

fn visible_time_span(rect: Rect, pixels_per_time: f32) -> u64 {
    ((rect.width() / pixels_per_time).ceil() as u64).max(1)
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

fn draw_insert_line(ui: &Ui, insert_index: usize, row_rects: &[Rect], width: f32, fallback_y: f32) {
    let anchor = row_rects
        .get(insert_index)
        .map(|rect| rect.top())
        .or_else(|| row_rects.last().map(|rect| rect.bottom()))
        .unwrap_or(fallback_y);
    let left = row_rects
        .first()
        .map(|rect| rect.left())
        .unwrap_or_else(|| ui.min_rect().left());
    ui.painter().line_segment(
        [pos2(left, anchor), pos2(left + width, anchor)],
        Stroke::new(3.0, Color32::from_rgb(255, 70, 70)),
    );
    ui.painter().add(Shape::convex_polygon(
        vec![
            pos2(left, anchor - 5.0),
            pos2(left, anchor + 5.0),
            pos2(left + 8.0, anchor),
        ],
        Color32::from_rgb(255, 70, 70),
        Stroke::NONE,
    ));
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

fn format_value_for_type(bits: &[u8], bit_offset: usize, ty: &WaveSignalType) -> String {
    match ty {
        WaveSignalType::Bool => {
            if get_bit(bits, bit_offset) {
                "true".to_owned()
            } else {
                "false".to_owned()
            }
        }
        &WaveSignalType::Int { signed, width } => format_int_value(bits, bit_offset, width, signed),
        WaveSignalType::Array { len, element } => {
            let stride = element.bit_len();
            let elements = (0..*len)
                .map(|index| format_value_for_type(bits, bit_offset + index * stride, element))
                .collect::<Vec<_>>();
            format!("[{}]", elements.join(", "))
        }
        WaveSignalType::Tuple(elements) => {
            let mut offset = bit_offset;
            let values = elements
                .iter()
                .map(|element| {
                    let value = format_value_for_type(bits, offset, element);
                    offset += element.bit_len();
                    value
                })
                .collect::<Vec<_>>();
            if values.len() == 1 {
                format!("({},)", values[0])
            } else {
                format!("({})", values.join(", "))
            }
        }
        WaveSignalType::Struct { name, fields } => {
            let mut offset = bit_offset;
            let values = fields
                .iter()
                .map(|(field_name, field_ty)| {
                    let value = format_value_for_type(bits, offset, field_ty);
                    offset += field_ty.bit_len();
                    format!("{field_name}={value}")
                })
                .collect::<Vec<_>>();
            format!("{name}.new({})", values.join(", "))
        }
        WaveSignalType::Enum { name, variants } => {
            let tag_width = enum_tag_width(variants.len());
            let tag = get_unsigned(bits, bit_offset, tag_width) as usize;
            let Some((variant_name, payload_ty)) = variants.get(tag) else {
                return format!("{name}.<invalid {tag}>");
            };
            match payload_ty {
                Some(payload_ty) => {
                    let payload = format_value_for_type(bits, bit_offset + tag_width, payload_ty);
                    format!("{name}.{variant_name}({payload})")
                }
                None => format!("{name}.{variant_name}"),
            }
        }
    }
}

fn format_int_value(bits: &[u8], bit_offset: usize, width: usize, signed: bool) -> String {
    if width == 0 {
        return "0".to_owned();
    }
    if width > 128 {
        return format!("0x{:x}...", get_unsigned(bits, bit_offset, 128));
    }
    let value = get_unsigned(bits, bit_offset, width);
    if signed && width < 128 && get_bit(bits, bit_offset + width - 1) {
        let signed_value = value as i128 - (1i128 << width);
        signed_value.to_string()
    } else {
        value.to_string()
    }
}

fn get_unsigned(bits: &[u8], bit_offset: usize, bit_len: usize) -> u128 {
    let mut value = 0u128;
    for i in 0..bit_len.min(128) {
        if get_bit(bits, bit_offset + i) {
            value |= 1u128 << i;
        }
    }
    value
}

fn get_bit(bits: &[u8], bit: usize) -> bool {
    bits.get(bit / 8).is_some_and(|byte| ((byte >> (bit % 8)) & 1) != 0)
}
