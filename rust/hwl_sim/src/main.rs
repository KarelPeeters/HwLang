mod consts;
mod format;
mod panels;
mod rows;
mod time;
mod widgets;

use crate::consts::{
    COLOR_CURSOR, COLOR_CURSOR_STATS_BORDER, COLOR_DRAG_INSERT, COLOR_GROUP_EVEN_BG, COLOR_GROUP_GUIDE_STROKE,
    COLOR_GROUP_HOVER_BG, COLOR_GROUP_ODD_BG, COLOR_GROUP_SELECTED_BG, COLOR_ROW_HOVER_BG, COLOR_ROW_SELECTED_BG,
    COLOR_SEPARATOR_STROKE, COLOR_SPACER_EVEN_BG, COLOR_SPACER_HOVER_BG, COLOR_SPACER_LINE, COLOR_SPACER_ODD_BG,
    COLOR_SPACER_SELECTED_BG, COLOR_STATUS_TEXT, COLOR_TEXT_MUTED, COLOR_TEXT_PRIMARY, COLOR_TEXT_STRONG,
    COLOR_WAVE_EVEN_BG, COLOR_WAVE_EXPANDED_BG, COLOR_WAVE_ODD_BG, COLOR_WAVE_SIGNAL_STROKE, COLOR_WAVE_SUBSECTION_BG,
    CURSOR_STATS_COLUMN_WIDTH, DEFAULT_ROW_LABEL_WIDTH, MAX_PIXELS_PER_TIME, MIN_PIXELS_PER_TIME, MIN_ROW_LABEL_WIDTH,
    ROW_HEIGHT,
};
use crate::format::{WaveRadix, enum_tag_width, format_value_for_type_with_radix, get_bit, get_unsigned};
use crate::panels::{draw_hierarchy, draw_signal_panel_row, filtered_signal_ids, filtered_signal_ids_for_module};
use crate::rows::{
    DropPlacement, DropTarget, RowDrag, VisibleWaveRow, VisibleWaveRowKind, WaveRow, WaveRowKey, WaveRowKind,
    best_drop_target_index, delete_selected_rows, drag_row_ids, drain_drag_rows_for_move, drop_placement,
    drop_targets_for_rows, empty_panel_area_clicked, group_blocks_are_contiguous, included_row_ids,
    placement_after_row, preserve_or_select_dragged_row, reparent_drag_roots, row_parent_map, selected_signal_row_ids,
    update_wave_row_selection, visible_wave_rows,
};
use crate::time::{
    ZoomDrag, clamp_time_view_start, draw_cursor, draw_cursor_stats_header, draw_dotted_cursor, draw_time_axis,
    draw_time_grid, draw_time_view_range, draw_zoom_selection, edge_counts, time_from_pointer, time_to_x,
    wave_content_width, zoom_to_selection,
};
use crate::widgets::draw_disclosure_icon;
use eframe::egui::scroll_area::ScrollBarVisibility;
use eframe::egui::{
    self, Align2, CentralPanel, Context, FontId, Key, Rect, ScrollArea, Sense, Shape, SidePanel, Stroke,
    TopBottomPanel, Ui, ViewportBuilder, pos2, vec2,
};
use hwl_language::sim::recorder::{WaveSignal, WaveSignalType, WaveStore};
use std::collections::{BTreeMap, BTreeSet};
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
    time_view_start: f32,
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
    last_group_name_click: Option<(u64, f64)>,
    row_drag: Option<RowDrag>,
    zoom_drag: Option<ZoomDrag>,
    cursor_dragging: bool,
    alt_cursor_pending: Option<u64>,
    secondary_cursor_time: Option<u64>,
    dragging_signals: Vec<usize>,
    collapsed_hierarchy: BTreeSet<String>,
    expanded_rows: BTreeSet<WaveRowKey>,
    selected_subsections: BTreeSet<WaveRowKey>,
    display_options: BTreeMap<WaveRowKey, WaveDisplayOptions>,
    context_menu: Option<WaveContextMenu>,
    show_ports: bool,
    show_signals: bool,
}

#[derive(Debug, Copy, Clone)]
struct WaveContextMenu {
    pos: egui::Pos2,
    key: Option<WaveRowKey>,
    placement: DropPlacement,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum WaveRenderMode {
    Digital,
    Analog,
}

#[derive(Debug, Copy, Clone)]
struct WaveDisplayOptions {
    radix: WaveRadix,
    render_mode: WaveRenderMode,
}

impl Default for WaveDisplayOptions {
    fn default() -> Self {
        Self {
            radix: WaveRadix::Dec,
            render_mode: WaveRenderMode::Digital,
        }
    }
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
            time_view_start: 0.0,
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
            last_group_name_click: None,
            row_drag: None,
            zoom_drag: None,
            cursor_dragging: false,
            alt_cursor_pending: None,
            secondary_cursor_time: None,
            dragging_signals: Vec::new(),
            collapsed_hierarchy: BTreeSet::new(),
            expanded_rows: BTreeSet::new(),
            selected_subsections: BTreeSet::new(),
            display_options: BTreeMap::new(),
            context_menu: None,
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
                            self.last_group_name_click = None;
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
                        if empty_panel_area_clicked(ui) {
                            self.selected_modules.clear();
                            self.last_selected_module = None;
                        }
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
            let load_requested = ui.button("Load").clicked()
                || path_response.lost_focus() && ui.input(|input| input.key_pressed(Key::Enter));
            if load_requested {
                self.load_store();
                ui.memory_mut(|memory| memory.surrender_focus(path_response.id));
            }
            ui.separator();
            ui.label("Zoom:");
            ui.add(
                egui::Slider::new(&mut self.pixels_per_time, MIN_PIXELS_PER_TIME..=MAX_PIXELS_PER_TIME)
                    .logarithmic(true),
            );
            ui.separator();
            ui.label("Time:");
            ui.add(egui::DragValue::new(&mut self.cursor_time).speed(1));
            ui.label("Alt-click/C over waves: Cursor2");
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
            ui.colored_label(COLOR_STATUS_TEXT, &self.status);
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
                self.pixels_per_time = (self.pixels_per_time * factor).min(MAX_PIXELS_PER_TIME);
            } else {
                self.pixels_per_time = (self.pixels_per_time / factor).max(MIN_PIXELS_PER_TIME);
            }
        }

        let max_time = store.max_time().max(1);
        let label_width = self.row_label_width;
        let visible_wave_width = (ui.available_width() - label_width).max(200.0);
        self.time_view_start = clamp_time_view_start(
            self.time_view_start,
            visible_wave_width / self.pixels_per_time,
            max_time,
        );
        self.cursor_time = self.cursor_time.min(max_time);
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
            debug_assert!(group_blocks_are_contiguous(&self.rows));
        }
        if !text_editing
            && ui.input(|input| input.key_pressed(Key::P) && (input.modifiers.ctrl || input.modifiers.command))
        {
            self.add_spacers_below_selected_signals();
        }
        if ui.input(|input| input.key_pressed(Key::Delete)) {
            if !self.selected_rows.is_empty() {
                delete_selected_rows(&mut self.rows, &self.selected_rows);
                debug_assert!(group_blocks_are_contiguous(&self.rows));
                self.expanded_rows
                    .retain(|key| self.rows.iter().any(|row| row.id == key.row_id));
                self.selected_subsections
                    .retain(|key| self.rows.iter().any(|row| row.id == key.row_id));
                self.display_options
                    .retain(|key, _| self.rows.iter().any(|row| row.id == key.row_id));
                self.selected_rows.clear();
                self.selected_row = None;
                self.last_selected_row = None;
            }
        }
        let pointer_released = ui.input(|input| input.pointer.any_released());
        if pointer_released {
            self.cursor_dragging = false;
        } else if !ui.input(|input| input.pointer.primary_down()) && self.row_drag.is_none() {
            self.dragging_signals.clear();
        }

        if self.rows.is_empty() && self.row_drag.is_none() && self.dragging_signals.is_empty() {
            return;
        }

        ScrollArea::vertical()
            .scroll_bar_visibility(ScrollBarVisibility::AlwaysHidden)
            .drag_to_scroll(false)
            .show(ui, |ui| {
                ui.set_min_width(label_width + visible_wave_width);
                draw_time_view_range(
                    ui,
                    label_width,
                    visible_wave_width,
                    &mut self.time_view_start,
                    self.pixels_per_time,
                    max_time,
                );
                self.time_view_start = clamp_time_view_start(
                    self.time_view_start,
                    visible_wave_width / self.pixels_per_time,
                    max_time,
                );
                let axis_rect = draw_time_axis(
                    ui,
                    self.pixels_per_time,
                    self.time_view_start,
                    max_time,
                    label_width,
                    visible_wave_width,
                );
                if let Some(secondary_cursor_time) = self.secondary_cursor_time {
                    if draw_cursor_stats_header(ui, axis_rect, self.cursor_time, secondary_cursor_time) {
                        self.secondary_cursor_time = None;
                        self.alt_cursor_pending = None;
                    }
                }
                ui.separator();
                let pointer_pos = ui.input(|input| input.pointer.interact_pos());
                let wave_gesture_active = ui.input(|input| {
                    input.modifiers.shift
                        && input.pointer.primary_down()
                        && input
                            .pointer
                            .press_origin()
                            .is_some_and(|origin| origin.x >= axis_rect.left())
                });
                let mut start_row_drag: Option<(usize, usize)> = None;
                let mut row_rects = Vec::new();
                let mut row_context_keys = Vec::new();
                let mut wave_bottom = axis_rect.bottom();
                let visible_rows = visible_wave_rows(&self.rows);
                for (visible_index, visible_row) in visible_rows.iter().enumerate() {
                    let row_selected = self.selected_rows.contains(&visible_row.row_id)
                        || self.selected_row == Some(visible_row.row_id);
                    match visible_row.kind {
                        VisibleWaveRowKind::Group => {
                            let dragging = self
                                .row_drag
                                .as_ref()
                                .is_some_and(|drag| drag.row_ids.contains(&visible_row.row_id));
                            let result = draw_group_row(
                                ui,
                                &mut self.rows[visible_row.row_index],
                                &mut self.last_group_name_click,
                                visible_index,
                                row_selected,
                                visible_row.depth,
                                label_width,
                                visible_wave_width,
                                dragging,
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
                            if !wave_gesture_active
                                && (result.label_drag_started || group_pointer_drag_started(ui, result.rect))
                                && self.row_drag.is_none()
                            {
                                preserve_or_select_dragged_row(
                                    visible_row.row_id,
                                    &mut self.selected_rows,
                                    &mut self.selected_row,
                                    &mut self.last_selected_row,
                                );
                                start_row_drag = Some((visible_row.row_index, visible_row.depth));
                            }
                            row_rects.push(result.rect);
                            row_context_keys.push(result.context_key);
                            wave_bottom = wave_bottom.max(result.rect.bottom());
                        }
                        VisibleWaveRowKind::Signal { signal_id } => {
                            if let Some(signal) = store.signals.get(signal_id) {
                                let dragging = self
                                    .row_drag
                                    .as_ref()
                                    .is_some_and(|drag| drag.row_ids.contains(&visible_row.row_id));
                                let result = draw_signal_rows(
                                    ui,
                                    &store,
                                    signal,
                                    visible_row.row_id,
                                    visible_index,
                                    row_selected,
                                    dragging,
                                    &self.expanded_rows,
                                    &self.selected_subsections,
                                    &self.display_options,
                                    self.cursor_time,
                                    self.secondary_cursor_time,
                                    self.pixels_per_time,
                                    self.time_view_start,
                                    max_time,
                                    visible_row.depth,
                                    label_width,
                                    visible_wave_width,
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
                                if let Some(time) = result.secondary_cursor_time {
                                    self.secondary_cursor_time = Some(time);
                                }
                                if result.cursor_drag_started {
                                    self.cursor_dragging = true;
                                }
                                if let Some(key) = result.clicked_key {
                                    update_wave_row_selection(
                                        ui,
                                        visible_index,
                                        &visible_rows,
                                        &mut self.selected_rows,
                                        &mut self.selected_row,
                                        &mut self.last_selected_row,
                                    );
                                    update_subsection_selection(ui, key, &mut self.selected_subsections);
                                } else if result.primary.clicked {
                                    self.selected_subsections.clear();
                                }
                                for key in result.expand_toggles {
                                    if !self.expanded_rows.remove(&key) {
                                        self.expanded_rows.insert(key);
                                    }
                                }
                                if !wave_gesture_active && result.primary.label_drag_started && self.row_drag.is_none()
                                {
                                    preserve_or_select_dragged_row(
                                        visible_row.row_id,
                                        &mut self.selected_rows,
                                        &mut self.selected_row,
                                        &mut self.last_selected_row,
                                    );
                                    start_row_drag = Some((visible_row.row_index, visible_row.depth));
                                }
                                row_rects.push(result.all_rect);
                                row_context_keys.push(result.context_key);
                                wave_bottom = wave_bottom.max(result.all_rect.bottom());
                            }
                        }
                        VisibleWaveRowKind::Spacer => {
                            let dragging = self
                                .row_drag
                                .as_ref()
                                .is_some_and(|drag| drag.row_ids.contains(&visible_row.row_id));
                            let result = draw_spacer_row(
                                ui,
                                visible_row.row_id,
                                visible_index,
                                row_selected,
                                dragging,
                                visible_row.depth,
                                label_width,
                                visible_wave_width,
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
                            if !wave_gesture_active && result.label_drag_started && self.row_drag.is_none() {
                                preserve_or_select_dragged_row(
                                    visible_row.row_id,
                                    &mut self.selected_rows,
                                    &mut self.selected_row,
                                    &mut self.last_selected_row,
                                );
                                start_row_drag = Some((visible_row.row_index, visible_row.depth));
                            }
                            row_rects.push(result.rect);
                            row_context_keys.push(result.context_key);
                            wave_bottom = wave_bottom.max(result.rect.bottom());
                        }
                    }
                }

                if let Some(pointer_pos) = pointer_pos {
                    if ui.input(|input| input.pointer.secondary_clicked()) {
                        let row_hit = row_rects.iter().position(|rect| rect.contains(pointer_pos));
                        let placement = row_hit
                            .map(|index| placement_after_row(&self.rows, visible_rows[index].row_index))
                            .unwrap_or_else(|| {
                                let targets =
                                    drop_targets_for_rows(&self.rows, &visible_rows, &row_rects, &BTreeSet::new());
                                let index = best_drop_target_index(pointer_pos, &targets, label_width);
                                targets.get(index).map(drop_placement).unwrap_or(DropPlacement {
                                    row_index: self.rows.len(),
                                    parent: None,
                                })
                            });
                        self.context_menu = Some(WaveContextMenu {
                            pos: pointer_pos,
                            key: row_hit.and_then(|index| row_context_keys[index]),
                            placement,
                        });
                    }
                }

                if let (Some(row_drag), Some(pointer_pos)) = (&mut self.row_drag, pointer_pos) {
                    let drop_targets = drop_targets_for_rows(&self.rows, &visible_rows, &row_rects, &row_drag.row_ids);
                    row_drag.target_index = best_drop_target_index(pointer_pos, &drop_targets, label_width);
                    row_drag.placement = drop_targets.get(row_drag.target_index).map(drop_placement);
                }
                let left_drop_targets = drop_targets_for_rows(&self.rows, &visible_rows, &row_rects, &BTreeSet::new());
                let left_drag_target_index = if !self.dragging_signals.is_empty() {
                    Some(pointer_pos.map_or(left_drop_targets.len(), |pos| {
                        best_drop_target_index(pos, &left_drop_targets, label_width)
                    }))
                } else {
                    None
                };
                if let Some((start_index, _drag_depth)) = start_row_drag {
                    if start_index < self.rows.len() {
                        let row_ids = drag_row_ids(&self.rows, start_index, &self.selected_rows);
                        let first_id = self.rows[start_index].id;
                        let drop_targets = drop_targets_for_rows(&self.rows, &visible_rows, &row_rects, &row_ids);
                        let target_index = pointer_pos.map_or(drop_targets.len(), |pos| {
                            best_drop_target_index(pos, &drop_targets, label_width)
                        });
                        let placement = drop_targets.get(target_index).map(drop_placement);
                        self.row_drag = Some(RowDrag {
                            row_ids,
                            first_id,
                            target_index,
                            placement,
                        });
                    }
                }

                if let Some(row_drag) = &self.row_drag {
                    draw_drag_name_boxes(ui, &visible_rows, &row_rects, &row_drag.row_ids, label_width);
                }

                let fallback_insert_y = axis_rect.bottom() + 7.0;
                if let Some(row_drag) = &self.row_drag {
                    let preview_targets =
                        drop_targets_for_rows(&self.rows, &visible_rows, &row_rects, &row_drag.row_ids);
                    draw_insert_line_for_targets(
                        ui,
                        row_drag.target_index,
                        &preview_targets,
                        label_width + visible_wave_width,
                        fallback_insert_y,
                    );
                } else if let Some(target_index) = left_drag_target_index {
                    draw_insert_line_for_targets(
                        ui,
                        target_index,
                        &left_drop_targets,
                        label_width + visible_wave_width,
                        fallback_insert_y,
                    );
                }
                if pointer_released && self.row_drag.is_some() {
                    if let Some(row_drag) = self.row_drag.take() {
                        let raw_insert_index = row_drag
                            .placement
                            .map(|placement| placement.row_index)
                            .unwrap_or(self.rows.len());
                        let new_parent = row_drag.placement.and_then(|placement| placement.parent);
                        let (mut rows, insert_index) =
                            drain_drag_rows_for_move(&mut self.rows, &row_drag.row_ids, raw_insert_index);
                        reparent_drag_roots(&mut rows, &row_drag.row_ids, new_parent);
                        self.rows.splice(insert_index..insert_index, rows);
                        debug_assert!(group_blocks_are_contiguous(&self.rows));
                        self.selected_row = Some(row_drag.first_id);
                        self.selected_rows.clear();
                        self.selected_rows.insert(row_drag.first_id);
                        self.last_selected_row = Some(row_drag.first_id);
                    }
                } else if pointer_released && !self.dragging_signals.is_empty() {
                    if ui.rect_contains_pointer(ui.max_rect()) {
                        let visible_rows = visible_wave_rows(&self.rows);
                        let drop_targets =
                            drop_targets_for_rows(&self.rows, &visible_rows, &row_rects, &BTreeSet::new());
                        let target =
                            left_drag_target_index.and_then(|target_index| drop_targets.get(target_index).copied());
                        let insert_index = target.map(|target| target.row_index).unwrap_or(self.rows.len());
                        let parent = target.and_then(|target| target.parent);
                        let new_rows = self
                            .dragging_signals
                            .clone()
                            .into_iter()
                            .map(|signal_id| self.make_signal_row(signal_id, parent))
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
                let splitter_response = ui.interact(
                    splitter_rect,
                    ui.make_persistent_id("wave-label-splitter"),
                    Sense::drag(),
                );
                if splitter_response.dragged() {
                    let delta_x = ui.input(|input| input.pointer.delta().x);
                    self.row_label_width = (self.row_label_width + delta_x).max(MIN_ROW_LABEL_WIDTH);
                }
                ui.painter().line_segment(
                    [
                        pos2(axis_rect.left(), axis_rect.top()),
                        pos2(axis_rect.left(), cursor_bottom),
                    ],
                    Stroke::new(1.0, COLOR_SEPARATOR_STROKE),
                );
                let (alt_down, primary_down, primary_pressed, primary_released, shift_down) = ui.input(|input| {
                    (
                        input.modifiers.alt,
                        input.pointer.primary_down(),
                        input.pointer.primary_pressed(),
                        input.pointer.primary_released(),
                        input.modifiers.shift,
                    )
                });
                if let Some(pointer_pos) = pointer_pos {
                    let alt_cursor_time = time_from_pointer(
                        axis_rect,
                        pointer_pos,
                        self.pixels_per_time,
                        self.time_view_start,
                        max_time,
                    );
                    if alt_down && primary_down && cursor_span.contains(pointer_pos) {
                        self.alt_cursor_pending = Some(alt_cursor_time);
                        self.secondary_cursor_time = Some(alt_cursor_time);
                    }
                    if alt_down && primary_pressed && cursor_span.contains(pointer_pos) {
                        self.secondary_cursor_time = Some(alt_cursor_time);
                    }
                    if primary_released {
                        if let Some(time) = self.alt_cursor_pending.take() {
                            self.secondary_cursor_time = Some(time);
                        }
                    }
                    if ui.input(|input| input.key_pressed(Key::C)) && cursor_span.contains(pointer_pos) {
                        self.secondary_cursor_time = Some(alt_cursor_time);
                    }
                }
                if let Some(pointer_pos) = pointer_pos {
                    if primary_down
                        && !shift_down
                        && !alt_down
                        && self.row_drag.is_none()
                        && pointer_pos.x >= axis_rect.left()
                        && cursor_span.contains(pointer_pos)
                    {
                        self.cursor_dragging = true;
                        self.cursor_time = time_from_pointer(
                            axis_rect,
                            pointer_pos,
                            self.pixels_per_time,
                            self.time_view_start,
                            max_time,
                        );
                    }
                    if primary_down && shift_down && self.row_drag.is_none() && cursor_span.contains(pointer_pos) {
                        let time = time_from_pointer(
                            axis_rect,
                            pointer_pos,
                            self.pixels_per_time,
                            self.time_view_start,
                            max_time,
                        );
                        if let Some(zoom_drag) = &mut self.zoom_drag {
                            zoom_drag.current_time = time;
                        } else {
                            self.zoom_drag = Some(ZoomDrag {
                                start_time: time,
                                current_time: time,
                            });
                        }
                        self.cursor_dragging = false;
                    }
                }
                if pointer_released {
                    if let Some(zoom_drag) = self.zoom_drag.take() {
                        let release_time = pointer_pos
                            .map(|pos| {
                                time_from_pointer(axis_rect, pos, self.pixels_per_time, self.time_view_start, max_time)
                            })
                            .unwrap_or(zoom_drag.current_time);
                        if let Some((time_view_start, pixels_per_time)) =
                            zoom_to_selection(visible_wave_width, max_time, zoom_drag.start_time, release_time)
                        {
                            self.time_view_start = time_view_start;
                            self.pixels_per_time = pixels_per_time;
                        }
                    }
                }
                if let Some(zoom_drag) = self.zoom_drag {
                    draw_zoom_selection(
                        ui.painter(),
                        cursor_span,
                        zoom_drag.start_time,
                        zoom_drag.current_time,
                        self.pixels_per_time,
                        self.time_view_start,
                    );
                }
                draw_cursor(
                    ui.painter(),
                    cursor_span,
                    self.cursor_time,
                    self.pixels_per_time,
                    self.time_view_start,
                );
                if let Some(secondary_cursor_time) = self.secondary_cursor_time {
                    draw_dotted_cursor(
                        ui.painter(),
                        cursor_span,
                        secondary_cursor_time,
                        self.pixels_per_time,
                        self.time_view_start,
                    );
                }
                self.show_wave_context_menu(ui.ctx());
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
                self.time_view_start = 0.0;
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
                self.last_group_name_click = None;
                self.row_drag = None;
                self.zoom_drag = None;
                self.cursor_dragging = false;
                self.alt_cursor_pending = None;
                self.secondary_cursor_time = None;
                self.dragging_signals.clear();
                self.collapsed_hierarchy.clear();
                self.expanded_rows.clear();
                self.selected_subsections.clear();
                self.display_options.clear();
                self.context_menu = None;
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
            if empty_panel_area_clicked(ui) {
                self.selected_signals.clear();
                self.last_selected_signal = None;
            }
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
            if empty_panel_area_clicked(ui) {
                self.selected_signals.clear();
                self.last_selected_signal = None;
            }
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
            if empty_panel_area_clicked(ui) {
                self.selected_signals.clear();
                self.last_selected_signal = None;
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
                let module_signals =
                    filtered_signal_ids_for_module(store, &module_path, self.show_ports, self.show_signals);
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

    fn make_signal_row(&mut self, signal_id: usize, parent: Option<u64>) -> WaveRow {
        WaveRow {
            id: self.next_row_id(),
            kind: WaveRowKind::Signal { signal_id, parent },
        }
    }

    fn make_group_row(&mut self, name: String, parent: Option<u64>) -> WaveRow {
        WaveRow {
            id: self.next_row_id(),
            kind: WaveRowKind::Group {
                name,
                collapsed: false,
                parent,
                editing: false,
            },
        }
    }

    fn make_spacer_row(&mut self, parent: Option<u64>) -> WaveRow {
        WaveRow {
            id: self.next_row_id(),
            kind: WaveRowKind::Spacer { parent },
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
        let group = self.make_group_row(name, None);
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
            let group = self.make_group_row("group".to_owned(), None);
            self.selected_rows.clear();
            self.selected_rows.insert(group.id);
            self.selected_row = Some(group.id);
            self.last_selected_row = Some(group.id);
            self.rows.push(group);
            return;
        }

        let selected = self.selected_rows.clone();
        let row_parents = row_parent_map(&self.rows);
        let included_ids = included_row_ids(&self.rows, &selected, &row_parents);
        let mut selected_indices = self
            .rows
            .iter()
            .enumerate()
            .filter_map(|(index, row)| included_ids.contains(&row.id).then_some(index))
            .collect::<Vec<_>>();
        if selected_indices.is_empty() {
            return;
        }
        selected_indices.sort_unstable();
        let original_insert_index = selected_indices[0];
        let new_parent = self.rows[original_insert_index].parent_id();
        let group = self.make_group_row("group".to_owned(), new_parent);
        let group_id = group.id;

        let selected_set = selected_indices.into_iter().collect::<BTreeSet<_>>();
        let mut grouped_rows = Vec::new();
        let mut kept_rows = Vec::new();
        let mut insert_index = 0;
        for (index, mut row) in self.rows.drain(..).enumerate() {
            if selected_set.contains(&index) {
                if !row.parent_id().is_some_and(|parent| included_ids.contains(&parent)) {
                    row.set_parent_id(Some(group_id));
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
        self.rows
            .splice(insert_index..insert_index, std::iter::once(group).chain(grouped_rows));
        self.selected_rows.clear();
        self.selected_rows.insert(group_id);
        self.selected_row = Some(group_id);
        self.last_selected_row = Some(group_id);
    }

    fn insert_group_at(&mut self, placement: DropPlacement) {
        let group = self.make_group_row("group".to_owned(), placement.parent);
        let group_id = group.id;
        let insert_index = placement.row_index.min(self.rows.len());
        self.rows.insert(insert_index, group);
        self.selected_rows.clear();
        self.selected_rows.insert(group_id);
        self.selected_row = Some(group_id);
        self.last_selected_row = Some(group_id);
        debug_assert!(group_blocks_are_contiguous(&self.rows));
    }

    fn insert_spacer_at(&mut self, placement: DropPlacement) {
        let spacer = self.make_spacer_row(placement.parent);
        let spacer_id = spacer.id;
        let insert_index = placement.row_index.min(self.rows.len());
        self.rows.insert(insert_index, spacer);
        self.selected_rows.clear();
        self.selected_rows.insert(spacer_id);
        self.selected_row = Some(spacer_id);
        self.last_selected_row = Some(spacer_id);
        debug_assert!(group_blocks_are_contiguous(&self.rows));
    }

    fn add_spacers_below_selected_signals(&mut self) {
        let selected_ids = self.selected_rows.clone();
        let mut targets = self
            .rows
            .iter()
            .enumerate()
            .filter_map(|(index, row)| {
                (selected_ids.contains(&row.id) && matches!(row.kind, WaveRowKind::Signal { .. }))
                    .then_some((index + 1, row.parent_id()))
            })
            .collect::<Vec<_>>();
        targets.sort_by_key(|(index, _)| *index);
        targets.dedup();

        let mut first_spacer = None;
        for (index, parent) in targets.into_iter().rev() {
            let spacer = self.make_spacer_row(parent);
            first_spacer = Some(spacer.id);
            self.rows.insert(index.min(self.rows.len()), spacer);
        }
        if let Some(first_spacer) = first_spacer {
            self.selected_rows.clear();
            self.selected_rows.insert(first_spacer);
            self.selected_row = Some(first_spacer);
            self.last_selected_row = Some(first_spacer);
        }
        debug_assert!(group_blocks_are_contiguous(&self.rows));
    }

    fn show_wave_context_menu(&mut self, ctx: &Context) {
        let Some(menu) = self.context_menu else {
            return;
        };
        let mut close = false;
        egui::Area::new(egui::Id::new("wave-context-menu"))
            .order(egui::Order::Foreground)
            .fixed_pos(menu.pos)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_min_width(190.0);
                    if let Some(key) = menu.key {
                        ui.label("Radix");
                        ui.horizontal(|ui| {
                            for (label, radix) in [
                                ("bin", WaveRadix::Bin),
                                ("hex", WaveRadix::Hex),
                                ("dec", WaveRadix::Dec),
                            ] {
                                if ui.button(label).clicked() {
                                    self.display_options.entry(key).or_default().radix = radix;
                                    close = true;
                                }
                            }
                        });
                        ui.label("Render");
                        ui.horizontal(|ui| {
                            for (label, render_mode) in
                                [("digital", WaveRenderMode::Digital), ("analog", WaveRenderMode::Analog)]
                            {
                                if ui.button(label).clicked() {
                                    self.display_options.entry(key).or_default().render_mode = render_mode;
                                    close = true;
                                }
                            }
                        });
                    } else {
                        ui.add_enabled(false, egui::Button::new("radix: bin / hex / dec"));
                        ui.add_enabled(false, egui::Button::new("render: digital / analog"));
                    }

                    ui.separator();
                    let has_selected_signal = selected_signal_row_ids(&self.rows, &self.selected_rows)
                        .next()
                        .is_some();
                    if ui
                        .add_enabled(has_selected_signal, egui::Button::new("group selected signals"))
                        .clicked()
                    {
                        self.create_group_from_selected_signals();
                        close = true;
                    }
                    if ui.button("create empty group below").clicked() {
                        self.insert_group_at(menu.placement);
                        close = true;
                    }
                    if ui.button("create spacer below").clicked() {
                        self.insert_spacer_at(menu.placement);
                        close = true;
                    }
                });
            });
        if close || ctx.input(|input| input.key_pressed(Key::Escape)) {
            self.context_menu = None;
        }
    }

    fn create_group_from_selected_signals(&mut self) {
        let signal_rows = selected_signal_row_ids(&self.rows, &self.selected_rows).collect::<BTreeSet<_>>();
        if signal_rows.is_empty() {
            return;
        }
        let old_selection = std::mem::replace(&mut self.selected_rows, signal_rows);
        self.create_group_from_selection();
        if self.selected_rows.is_empty() {
            self.selected_rows = old_selection;
        }
    }
}

#[derive(Clone)]
struct RowResult {
    clicked: bool,
    label_drag_started: bool,
    cursor_time: Option<u64>,
    secondary_cursor_time: Option<u64>,
    cursor_drag_started: bool,
    expand_toggles: Vec<WaveRowKey>,
    clicked_key: Option<WaveRowKey>,
    context_key: Option<WaveRowKey>,
    rect: Rect,
    label_rect: Rect,
}

struct SignalRowsResult {
    primary: RowResult,
    all_rect: Rect,
    all_label_rect: Rect,
    cursor_time: Option<u64>,
    secondary_cursor_time: Option<u64>,
    cursor_drag_started: bool,
    expand_toggles: Vec<WaveRowKey>,
    clicked_key: Option<WaveRowKey>,
    context_key: Option<WaveRowKey>,
}

fn update_subsection_selection(ui: &Ui, key: WaveRowKey, selected_subsections: &mut BTreeSet<WaveRowKey>) {
    let modifiers = ui.input(|input| input.modifiers);
    if modifiers.ctrl || modifiers.command {
        if !selected_subsections.remove(&key) {
            selected_subsections.insert(key);
        }
    } else {
        selected_subsections.clear();
        selected_subsections.insert(key);
    }
}

fn draw_group_row(
    ui: &mut Ui,
    row: &mut WaveRow,
    last_group_name_click: &mut Option<(u64, f64)>,
    row_index: usize,
    selected: bool,
    depth: usize,
    label_width: f32,
    wave_width: f32,
    dragging: bool,
) -> RowResult {
    let (row_rect, _) = ui.allocate_exact_size(vec2(label_width + wave_width, ROW_HEIGHT), Sense::hover());
    let row_response = ui.interact(
        row_rect,
        ui.make_persistent_id(("group-row", row.id)),
        Sense::click_and_drag(),
    );
    let label_rect = Rect::from_min_size(row_rect.min, vec2(label_width, ROW_HEIGHT));
    let indent = depth as f32 * 18.0;
    let icon_hit_rect = Rect::from_min_size(
        pos2(label_rect.left() + indent, label_rect.top()),
        vec2(28.0, ROW_HEIGHT),
    );
    let icon_rect = Rect::from_center_size(icon_hit_rect.center(), vec2(12.0, 12.0));
    let name_rect = Rect::from_min_max(
        pos2(icon_hit_rect.right(), label_rect.top() + 3.0),
        label_rect.right_bottom(),
    );
    let painter = ui.painter_at(row_rect);
    let click_pos = if row_response.clicked() {
        row_response.interact_pointer_pos()
    } else {
        None
    };
    let bg = if selected && !dragging {
        COLOR_GROUP_SELECTED_BG
    } else if row_response.hovered() {
        COLOR_GROUP_HOVER_BG
    } else if row_index % 2 == 0 {
        COLOR_GROUP_EVEN_BG
    } else {
        COLOR_GROUP_ODD_BG
    };
    painter.rect_filled(label_rect, 0.0, bg);
    draw_group_guides(ui.painter(), label_rect, depth);

    if let WaveRowKind::Group {
        name,
        collapsed,
        editing,
        ..
    } = &mut row.kind
    {
        let icon_clicked = click_pos.is_some_and(|pos| icon_hit_rect.contains(pos));
        let name_clicked = click_pos.is_some_and(|pos| name_rect.contains(pos));
        if icon_clicked {
            *collapsed = !*collapsed;
        }
        draw_disclosure_icon(ui.painter(), icon_rect, !*collapsed);
        let edit_rect = name_rect;
        let name_id = ui.make_persistent_id(("group-name", row.id));
        if name_clicked {
            let now = ui.input(|input| input.time);
            let repeated_click =
                last_group_name_click.is_some_and(|(last_id, last_time)| last_id == row.id && now - last_time <= 0.45);
            if row_response.double_clicked() || repeated_click {
                *editing = true;
                *last_group_name_click = None;
                ui.memory_mut(|memory| memory.request_focus(name_id));
            } else {
                *last_group_name_click = Some((row.id, now));
            }
        }
        if *editing {
            let was_focused = ui.memory(|memory| memory.has_focus(name_id));
            let response = ui.put(
                edit_rect,
                egui::TextEdit::singleline(name)
                    .font(FontId::proportional(13.0))
                    .id(name_id)
                    .frame(true),
            );
            if !was_focused && !response.has_focus() {
                response.request_focus();
            }
            if !was_focused {
                select_all_text_edit(ui, response.id, name.chars().count());
            }
            if response.lost_focus()
                || ui.input(|input| input.key_pressed(Key::Enter) || input.key_pressed(Key::Escape))
            {
                *editing = false;
            }
        } else {
            ui.painter().text(
                pos2(edit_rect.left(), edit_rect.center().y),
                Align2::LEFT_CENTER,
                name,
                FontId::proportional(13.0),
                COLOR_TEXT_PRIMARY,
            );
        }
    }
    RowResult {
        clicked: row_response.clicked(),
        label_drag_started: row_response.drag_started() || row_response.dragged(),
        cursor_time: None,
        secondary_cursor_time: None,
        cursor_drag_started: false,
        expand_toggles: Vec::new(),
        clicked_key: None,
        context_key: None,
        rect: row_rect,
        label_rect,
    }
}

fn draw_group_guides(painter: &egui::Painter, label_rect: Rect, depth: usize) {
    for level in 0..depth {
        let x = label_rect.left() + 12.0 + level as f32 * 18.0;
        painter.line_segment(
            [pos2(x, label_rect.top() - 1.0), pos2(x, label_rect.bottom() + 1.0)],
            Stroke::new(2.0, COLOR_GROUP_GUIDE_STROKE),
        );
    }
}

fn draw_spacer_row(
    ui: &mut Ui,
    row_id: u64,
    row_index: usize,
    selected: bool,
    dragging: bool,
    depth: usize,
    label_width: f32,
    wave_width: f32,
) -> RowResult {
    let (row_rect, _) = ui.allocate_exact_size(vec2(label_width + wave_width, ROW_HEIGHT), Sense::hover());
    let label_rect = Rect::from_min_size(row_rect.min, vec2(label_width, ROW_HEIGHT));
    let wave_rect = Rect::from_min_size(
        pos2(row_rect.left() + label_width, row_rect.top()),
        vec2(wave_width, ROW_HEIGHT),
    );
    let label_response = ui.interact(
        label_rect,
        ui.make_persistent_id(("spacer-row", row_id)),
        Sense::click_and_drag(),
    );
    let wave_response = ui.interact(
        wave_rect,
        ui.make_persistent_id(("spacer-wave", row_id)),
        Sense::click(),
    );
    let bg = if selected && !dragging {
        COLOR_SPACER_SELECTED_BG
    } else if label_response.hovered() || wave_response.hovered() {
        COLOR_SPACER_HOVER_BG
    } else if row_index % 2 == 0 {
        COLOR_SPACER_EVEN_BG
    } else {
        COLOR_SPACER_ODD_BG
    };
    let painter = ui.painter_at(row_rect);
    painter.rect_filled(row_rect, 0.0, bg);
    draw_group_guides(ui.painter(), label_rect, depth);
    painter.line_segment(
        [
            pos2(row_rect.left() + depth as f32 * 18.0 + 24.0, row_rect.center().y),
            pos2(row_rect.right(), row_rect.center().y),
        ],
        Stroke::new(1.0, COLOR_SPACER_LINE),
    );

    RowResult {
        clicked: label_response.clicked() || wave_response.clicked(),
        label_drag_started: label_response.drag_started(),
        cursor_time: None,
        secondary_cursor_time: None,
        cursor_drag_started: false,
        expand_toggles: Vec::new(),
        clicked_key: None,
        context_key: None,
        rect: row_rect,
        label_rect,
    }
}

fn draw_drag_name_boxes(
    ui: &Ui,
    visible_rows: &[VisibleWaveRow],
    row_rects: &[Rect],
    dragged_ids: &BTreeSet<u64>,
    label_width: f32,
) {
    let mut merged: Vec<Rect> = Vec::new();
    let mut current: Option<Rect> = None;
    for (visible_row, row_rect) in visible_rows.iter().zip(row_rects.iter().copied()) {
        if !dragged_ids.contains(&visible_row.row_id) {
            if let Some(rect) = current.take() {
                merged.push(rect);
            }
            continue;
        }
        let rect = Rect::from_min_max(row_rect.min, pos2(row_rect.left() + label_width, row_rect.bottom()));
        current = Some(current.map_or(rect, |last| last.union(rect)));
    }
    if let Some(rect) = current {
        merged.push(rect);
    }
    if merged.is_empty() {
        return;
    }

    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        ui.make_persistent_id("drag-name-boxes"),
    ));
    for rect in merged {
        let rect = rect.shrink2(vec2(4.0, 2.0));
        painter.rect_stroke(rect, 3.0, Stroke::new(2.0, COLOR_DRAG_INSERT));
    }
}

fn select_all_text_edit(ui: &Ui, id: egui::Id, char_count: usize) {
    let mut state = egui::widgets::text_edit::TextEditState::load(ui.ctx(), id).unwrap_or_default();
    state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
        egui::text::CCursor::new(0),
        egui::text::CCursor::new(char_count),
    )));
    state.store(ui.ctx(), id);
}

fn group_pointer_drag_started(ui: &Ui, rect: Rect) -> bool {
    ui.input(|input| {
        if !input.pointer.primary_down() {
            return false;
        }
        let Some(origin) = input.pointer.press_origin() else {
            return false;
        };
        if !rect.contains(origin) {
            return false;
        }
        input
            .pointer
            .interact_pos()
            .is_some_and(|pos| pos.distance(origin) > 3.0)
    })
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
    selected_subsections: &BTreeSet<WaveRowKey>,
    display_options: &BTreeMap<WaveRowKey, WaveDisplayOptions>,
    cursor_time: u64,
    secondary_cursor_time: Option<u64>,
    pixels_per_time: f32,
    time_view_start: f32,
    max_time: u64,
    group_depth: usize,
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
        group_depth,
        expanded_rows,
        selected_subsections,
        display_options,
        cursor_time,
        secondary_cursor_time,
        pixels_per_time,
        time_view_start,
        max_time,
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
    selected_subsections: &BTreeSet<WaveRowKey>,
    display_options: &BTreeMap<WaveRowKey, WaveDisplayOptions>,
    cursor_time: u64,
    secondary_cursor_time: Option<u64>,
    pixels_per_time: f32,
    time_view_start: f32,
    max_time: u64,
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
        dragging,
        is_composite(ty),
        expanded,
        key,
        can_reorder,
        depth,
        selected_subsections.contains(&key),
        display_options.get(&key).copied().unwrap_or_default(),
        cursor_time,
        secondary_cursor_time,
        pixels_per_time,
        time_view_start,
        max_time,
        label_width,
        wave_width,
    );
    let mut result = SignalRowsResult {
        primary: leaf.clone(),
        all_rect: leaf.rect,
        all_label_rect: leaf.label_rect,
        cursor_time: leaf.cursor_time,
        secondary_cursor_time: leaf.secondary_cursor_time,
        cursor_drag_started: leaf.cursor_drag_started,
        expand_toggles: leaf.expand_toggles,
        clicked_key: leaf.clicked_key,
        context_key: leaf.context_key,
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
                dragging,
                false,
                depth + 1,
                expanded_rows,
                selected_subsections,
                display_options,
                cursor_time,
                secondary_cursor_time,
                pixels_per_time,
                time_view_start,
                max_time,
                label_width,
                wave_width,
            );
            result.all_rect = result.all_rect.union(child.all_rect);
            result.all_label_rect = result.all_label_rect.union(child.all_label_rect);
            if result.cursor_time.is_none() {
                result.cursor_time = child.cursor_time;
            }
            if result.secondary_cursor_time.is_none() {
                result.secondary_cursor_time = child.secondary_cursor_time;
            }
            result.cursor_drag_started |= child.cursor_drag_started;
            result.expand_toggles.extend(child.expand_toggles);
            if result.clicked_key.is_none() {
                result.clicked_key = child.clicked_key;
            }
            if result.context_key.is_none() {
                result.context_key = child.context_key;
            }
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
    subsection_selected: bool,
    display_options: WaveDisplayOptions,
    cursor_time: u64,
    secondary_cursor_time: Option<u64>,
    pixels_per_time: f32,
    time_view_start: f32,
    max_time: u64,
    label_width: f32,
    wave_width: f32,
) -> RowResult {
    let (row_rect, _) = ui.allocate_exact_size(vec2(label_width + wave_width, ROW_HEIGHT), Sense::hover());
    let label_rect = Rect::from_min_size(row_rect.min, vec2(label_width, ROW_HEIGHT));
    let content_wave_width = wave_content_width(time_view_start, pixels_per_time, wave_width, max_time);
    let wave_rect = Rect::from_min_size(
        pos2(row_rect.left() + label_width, row_rect.top()),
        vec2(content_wave_width, ROW_HEIGHT),
    );
    let label_response = ui.interact(
        label_rect,
        ui.make_persistent_id(("row-label", key)),
        Sense::click_and_drag(),
    );
    let wave_response = ui.interact(
        wave_rect,
        ui.make_persistent_id(("row-wave", key)),
        Sense::click_and_drag(),
    );
    let painter = ui.painter_at(row_rect);

    let bg = if selected && !dragging {
        COLOR_ROW_SELECTED_BG
    } else if subsection_selected {
        COLOR_WAVE_SUBSECTION_BG
    } else if expandable && expanded {
        COLOR_WAVE_EXPANDED_BG
    } else if label_response.hovered() || wave_response.hovered() {
        COLOR_ROW_HOVER_BG
    } else if row_index % 2 == 0 {
        COLOR_WAVE_EVEN_BG
    } else {
        COLOR_WAVE_ODD_BG
    };
    painter.rect_filled(row_rect, 0.0, bg);
    draw_group_guides(ui.painter(), label_rect, depth);

    let value = store
        .signal_value_at(signal.id, cursor_time)
        .map(|bits| format_value_for_type_with_radix(bits, key.bit_offset, ty, display_options.radix))
        .unwrap_or_else(|| "x".to_owned());
    let cursor_stats = secondary_cursor_time.map(|secondary| {
        edge_counts(
            &store.changes[signal.id],
            key.bit_offset,
            key.bit_len,
            cursor_time,
            secondary,
        )
    });
    let stats_text = cursor_stats.map(|stats| format!("↑{} ↓{} ↕{}", stats.posedges, stats.negedges, stats.toggles));
    let icon_rect = Rect::from_min_size(
        pos2(
            label_rect.left() + 4.0 + depth as f32 * 18.0,
            label_rect.center().y - 6.0,
        ),
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
    let label_text_rect = if stats_text.is_some() {
        Rect::from_min_max(
            label_rect.min,
            pos2(
                (label_rect.right() - CURSOR_STATS_COLUMN_WIDTH).max(label_rect.left()),
                label_rect.bottom(),
            ),
        )
    } else {
        label_rect
    };
    painter.with_clip_rect(label_text_rect).text(
        pos2(icon_rect.right() + 4.0, label_rect.center().y),
        Align2::LEFT_CENTER,
        format!("{label} = {value}"),
        FontId::proportional(13.0),
        COLOR_TEXT_STRONG,
    );
    if let Some(stats_text) = stats_text {
        let stats_rect = Rect::from_min_max(
            pos2(
                (label_rect.right() - CURSOR_STATS_COLUMN_WIDTH).max(label_rect.left()),
                label_rect.top(),
            ),
            label_rect.right_bottom(),
        );
        painter.line_segment(
            [stats_rect.left_top(), stats_rect.left_bottom()],
            Stroke::new(1.0, COLOR_CURSOR_STATS_BORDER),
        );
        painter.with_clip_rect(stats_rect).text(
            stats_rect.center(),
            Align2::CENTER_CENTER,
            stats_text,
            FontId::monospace(12.0),
            COLOR_CURSOR,
        );
    }
    draw_waveform(
        &painter,
        wave_rect,
        &store.changes[signal.id],
        key.bit_offset,
        ty,
        display_options,
        pixels_per_time,
        time_view_start,
        max_time,
    );
    let alt_down = ui.input(|input| input.modifiers.alt);
    let cursor_time = wave_response
        .interact_pointer_pos()
        .filter(|pos| {
            !alt_down
                && !ui.input(|input| input.modifiers.shift)
                && (wave_response.clicked() || wave_response.dragged())
                && wave_rect.contains(*pos)
        })
        .map(|pos| time_from_pointer(wave_rect, pos, pixels_per_time, time_view_start, max_time));
    let secondary_cursor_time = wave_response
        .interact_pointer_pos()
        .filter(|pos| alt_down && wave_response.clicked() && wave_rect.contains(*pos))
        .map(|pos| time_from_pointer(wave_rect, pos, pixels_per_time, time_view_start, max_time));
    let cursor_drag_started = !alt_down && !ui.input(|input| input.modifiers.shift) && wave_response.drag_started();
    let expand_toggles = icon_response
        .filter(|response| response.clicked())
        .map(|_| vec![key])
        .unwrap_or_default();

    RowResult {
        clicked: label_response.clicked() || wave_response.clicked(),
        label_drag_started: can_reorder && label_response.drag_started(),
        cursor_time,
        secondary_cursor_time,
        cursor_drag_started,
        expand_toggles,
        clicked_key: (!can_reorder && (label_response.clicked() || wave_response.clicked())).then_some(key),
        context_key: (label_response.secondary_clicked() || wave_response.secondary_clicked()).then_some(key),
        rect: row_rect,
        label_rect,
    }
}

fn draw_waveform(
    painter: &egui::Painter,
    rect: Rect,
    changes: &[hwl_language::sim::recorder::WaveChange],
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
    changes: &[hwl_language::sim::recorder::WaveChange],
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

fn draw_insert_line_for_targets(ui: &Ui, target_index: usize, targets: &[DropTarget], width: f32, fallback_y: f32) {
    let target = targets.get(target_index);
    let anchor = target
        .map(|target| target.y)
        .unwrap_or_else(|| targets.last().map(|target| target.y).unwrap_or(fallback_y));
    let base_left = target
        .or_else(|| targets.last())
        .map(|target| target.rect.left())
        .unwrap_or_else(|| ui.min_rect().left());
    let depth = target
        .or_else(|| targets.last())
        .map(|target| target.depth)
        .unwrap_or(0);
    let left = base_left + depth as f32 * 18.0;
    let right = base_left + width;
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        ui.make_persistent_id("insert-preview"),
    ));
    painter.line_segment(
        [pos2(left, anchor), pos2(right, anchor)],
        Stroke::new(3.0, COLOR_DRAG_INSERT),
    );
    painter.add(Shape::convex_polygon(
        vec![
            pos2(left, anchor - 5.0),
            pos2(left, anchor + 5.0),
            pos2(left + 8.0, anchor),
        ],
        COLOR_DRAG_INSERT,
        Stroke::NONE,
    ));
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
            result.push((
                ".tag".to_owned(),
                WaveSignalType::Int {
                    signed: false,
                    width: tag_width,
                },
                0,
                tag_width,
            ));
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

#[cfg(test)]
mod tests {
    use super::WaveGuiApp;
    use crate::rows::WaveRowKind;
    use std::collections::BTreeSet;

    #[test]
    fn spacers_are_inserted_below_each_selected_signal() {
        let mut app = WaveGuiApp::default();
        let signal_a = app.make_signal_row(0, None);
        let signal_b = app.make_signal_row(1, None);
        let signal_a_id = signal_a.id;
        let signal_b_id = signal_b.id;
        app.rows = vec![signal_a, signal_b];
        app.selected_rows = BTreeSet::from([signal_a_id, signal_b_id]);

        app.add_spacers_below_selected_signals();

        assert!(matches!(app.rows[0].kind, WaveRowKind::Signal { .. }));
        assert!(matches!(app.rows[1].kind, WaveRowKind::Spacer { parent: None }));
        assert!(matches!(app.rows[2].kind, WaveRowKind::Signal { .. }));
        assert!(matches!(app.rows[3].kind, WaveRowKind::Spacer { parent: None }));
    }
}
