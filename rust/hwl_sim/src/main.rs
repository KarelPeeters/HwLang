mod consts;
mod format;
mod panels;
mod render;
mod row_render;
mod rows;
mod time;
mod widgets;

use crate::consts::{
    COLOR_SEPARATOR_STROKE, COLOR_STATUS_TEXT, DEFAULT_ROW_LABEL_WIDTH, MAX_PIXELS_PER_TIME, MIN_PIXELS_PER_TIME,
    MIN_ROW_LABEL_WIDTH,
};
use crate::format::WaveRadix;
use crate::panels::{draw_hierarchy, draw_signal_panel_row, filtered_signal_ids, filtered_signal_ids_for_module};
use crate::render::{WaveDisplayOptions, WaveRenderMode};
use crate::row_render::{
    draw_drag_name_boxes, draw_group_row, draw_insert_line_for_targets, draw_signal_rows, draw_spacer_row,
    group_pointer_drag_started, update_subsection_selection,
};
use crate::rows::{
    DropPlacement, RowDrag, VisibleWaveRowKind, WaveRow, WaveRowKey, WaveRowKind, best_drop_target_index,
    delete_selected_rows, drag_row_ids, drain_drag_rows_for_move, drop_placement, drop_targets_for_rows,
    empty_panel_area_clicked, group_blocks_are_contiguous, included_row_ids, placement_after_row,
    preserve_or_select_dragged_row, reparent_drag_roots, row_parent_map, selected_signal_row_ids,
    update_wave_row_selection, visible_wave_rows,
};
use crate::time::{
    ZoomDrag, clamp_time_view_start, draw_cursor, draw_cursor_stats_header, draw_dotted_cursor, draw_time_axis,
    draw_time_view_range, draw_zoom_selection, time_from_pointer, zoom_to_selection,
};
use eframe::egui::scroll_area::ScrollBarVisibility;
use eframe::egui::{
    self, CentralPanel, Context, Key, Rect, ScrollArea, Sense, SidePanel, Stroke, TopBottomPanel, Ui, ViewportBuilder,
    pos2,
};
use hwl_language::sim::recorder::WaveStore;
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
