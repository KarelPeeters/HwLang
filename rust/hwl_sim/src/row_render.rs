use crate::consts::{
    COLOR_CURSOR, COLOR_CURSOR_STATS_BORDER, COLOR_DRAG_INSERT, COLOR_GROUP_EVEN_BG, COLOR_GROUP_GUIDE_STROKE,
    COLOR_GROUP_HOVER_BG, COLOR_GROUP_ODD_BG, COLOR_GROUP_SELECTED_BG, COLOR_ROW_HOVER_BG, COLOR_ROW_SELECTED_BG,
    COLOR_SPACER_EVEN_BG, COLOR_SPACER_HOVER_BG, COLOR_SPACER_LINE, COLOR_SPACER_ODD_BG, COLOR_SPACER_SELECTED_BG,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_STRONG, COLOR_WAVE_EVEN_BG, COLOR_WAVE_EXPANDED_BG, COLOR_WAVE_ODD_BG,
    COLOR_WAVE_SUBSECTION_BG, CURSOR_STATS_COLUMN_WIDTH, ROW_HEIGHT,
};
use crate::format::format_value_for_type_with_radix;
use crate::render::{WaveDisplayOptions, draw_waveform};
use crate::rows::{DropTarget, VisibleWaveRow, WaveRow, WaveRowKey, WaveRowKind};
use crate::state::SelectionState;
use crate::time::{edge_counts, time_from_pointer, wave_content_width};
use crate::type_layout::composite_children;
use crate::widgets::draw_disclosure_icon;
use eframe::egui::{self, Align2, FontId, Key, Rect, Sense, Shape, Stroke, Ui, pos2, vec2};
use hwl_language::sim::recorder::{WaveSignal, WaveSignalType, WaveStore};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
pub struct RowResult {
    pub clicked: bool,
    pub label_drag_started: bool,
    pub cursor_time: Option<u64>,
    pub secondary_cursor_time: Option<u64>,
    pub cursor_drag_started: bool,
    pub expand_toggles: Vec<WaveRowKey>,
    pub clicked_key: Option<WaveRowKey>,
    pub context_key: Option<WaveRowKey>,
    pub rect: Rect,
    pub label_rect: Rect,
}

pub struct SignalRowsResult {
    pub primary: RowResult,
    pub all_rect: Rect,
    pub all_label_rect: Rect,
    pub cursor_time: Option<u64>,
    pub secondary_cursor_time: Option<u64>,
    pub cursor_drag_started: bool,
    pub expand_toggles: Vec<WaveRowKey>,
    pub clicked_key: Option<WaveRowKey>,
    pub context_key: Option<WaveRowKey>,
}

#[derive(Clone, Copy)]
pub struct WaveViewport {
    pub pixels_per_time: f32,
    pub time_view_start: f32,
    pub max_time: u64,
    pub label_width: f32,
    pub wave_width: f32,
}

pub struct SignalRenderContext<'a> {
    pub store: &'a WaveStore,
    pub expanded_rows: &'a BTreeSet<WaveRowKey>,
    pub selected_subsections: &'a BTreeSet<WaveRowKey>,
    pub display_options: &'a BTreeMap<WaveRowKey, WaveDisplayOptions>,
    pub cursor_time: u64,
    pub secondary_cursor_time: Option<u64>,
    pub viewport: WaveViewport,
}

#[derive(Clone, Copy)]
pub struct SignalRowState {
    pub row_id: u64,
    pub row_index: usize,
    pub selected: bool,
    pub dragging: bool,
    pub group_depth: usize,
}

#[derive(Clone, Copy)]
struct SignalTreeState {
    row_index: usize,
    selected: bool,
    dragging: bool,
    can_reorder: bool,
    depth: usize,
}

#[derive(Clone, Copy)]
struct SignalLeafState {
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
}

pub fn update_subsection_selection(ui: &Ui, key: WaveRowKey, selected_subsections: &mut SelectionState<WaveRowKey>) {
    let modifiers = ui.input(|input| input.modifiers);
    selected_subsections.apply_single_selection(key, modifiers);
}

pub fn draw_group_row(
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

pub fn draw_spacer_row(
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

pub fn draw_drag_name_boxes(
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

pub fn group_pointer_drag_started(ui: &Ui, rect: Rect) -> bool {
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

pub fn draw_signal_rows(
    ui: &mut Ui,
    signal: &WaveSignal,
    row: SignalRowState,
    ctx: &SignalRenderContext<'_>,
) -> SignalRowsResult {
    let key = WaveRowKey {
        row_id: row.row_id,
        signal_id: signal.id,
        bit_offset: 0,
        bit_len: signal.bit_len,
        part: 0,
    };
    draw_signal_tree(
        ui,
        signal,
        &format!("{}.{}", signal.path.join("."), signal.name),
        &signal.ty,
        key,
        SignalTreeState {
            row_index: row.row_index,
            selected: row.selected,
            dragging: row.dragging,
            can_reorder: true,
            depth: row.group_depth,
        },
        ctx,
    )
}

fn draw_signal_tree(
    ui: &mut Ui,
    signal: &WaveSignal,
    label: &str,
    ty: &WaveSignalType,
    key: WaveRowKey,
    state: SignalTreeState,
    ctx: &SignalRenderContext<'_>,
) -> SignalRowsResult {
    let expanded = is_composite(ty) && ctx.expanded_rows.contains(&key);
    let leaf = draw_signal_leaf(
        ui,
        signal,
        label,
        ty,
        SignalLeafState {
            row_index: state.row_index,
            selected: state.selected && state.can_reorder,
            dragging: state.dragging,
            expandable: is_composite(ty),
            expanded,
            key,
            can_reorder: state.can_reorder,
            depth: state.depth,
            subsection_selected: ctx.selected_subsections.contains(&key),
            display_options: ctx.display_options.get(&key).copied().unwrap_or_default(),
        },
        ctx,
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
        for (child_index, child_info) in composite_children(ty).into_iter().enumerate() {
            let child_key = WaveRowKey {
                signal_id: signal.id,
                row_id: key.row_id,
                bit_offset: key.bit_offset + child_info.bit_offset,
                bit_len: child_info.bit_len.max(child_info.ty.bit_len()),
                part: key.part.wrapping_mul(131).wrapping_add(child_index as u64 + 1),
            };
            let child = draw_signal_tree(
                ui,
                signal,
                &child_info.name,
                &child_info.ty,
                child_key,
                SignalTreeState {
                    row_index: state.row_index,
                    selected: false,
                    dragging: state.dragging,
                    can_reorder: false,
                    depth: state.depth + 1,
                },
                ctx,
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
    signal: &WaveSignal,
    label: &str,
    ty: &WaveSignalType,
    state: SignalLeafState,
    ctx: &SignalRenderContext<'_>,
) -> RowResult {
    let viewport = ctx.viewport;
    let (row_rect, _) = ui.allocate_exact_size(
        vec2(viewport.label_width + viewport.wave_width, ROW_HEIGHT),
        Sense::hover(),
    );
    let label_rect = Rect::from_min_size(row_rect.min, vec2(viewport.label_width, ROW_HEIGHT));
    let content_wave_width = wave_content_width(
        viewport.time_view_start,
        viewport.pixels_per_time,
        viewport.wave_width,
        viewport.max_time,
    );
    let wave_rect = Rect::from_min_size(
        pos2(row_rect.left() + viewport.label_width, row_rect.top()),
        vec2(content_wave_width, ROW_HEIGHT),
    );
    let label_response = ui.interact(
        label_rect,
        ui.make_persistent_id(("row-label", state.key)),
        Sense::click_and_drag(),
    );
    let wave_response = ui.interact(
        wave_rect,
        ui.make_persistent_id(("row-wave", state.key)),
        Sense::click_and_drag(),
    );
    let painter = ui.painter_at(row_rect);

    let bg = if state.selected && !state.dragging {
        COLOR_ROW_SELECTED_BG
    } else if state.subsection_selected {
        COLOR_WAVE_SUBSECTION_BG
    } else if state.expandable && state.expanded {
        COLOR_WAVE_EXPANDED_BG
    } else if label_response.hovered() || wave_response.hovered() {
        COLOR_ROW_HOVER_BG
    } else if state.row_index % 2 == 0 {
        COLOR_WAVE_EVEN_BG
    } else {
        COLOR_WAVE_ODD_BG
    };
    painter.rect_filled(row_rect, 0.0, bg);
    draw_group_guides(ui.painter(), label_rect, state.depth);

    let value = ctx
        .store
        .signal_value_at(signal.id, ctx.cursor_time)
        .map(|bits| format_value_for_type_with_radix(bits, state.key.bit_offset, ty, state.display_options.radix))
        .unwrap_or_else(|| "x".to_owned());
    let cursor_stats = ctx.secondary_cursor_time.map(|secondary| {
        edge_counts(
            &ctx.store.changes[signal.id],
            state.key.bit_offset,
            state.key.bit_len,
            ctx.cursor_time,
            secondary,
        )
    });
    let stats_text = cursor_stats.map(|stats| format!("↑{} ↓{} ↕{}", stats.posedges, stats.negedges, stats.toggles));
    let icon_rect = Rect::from_min_size(
        pos2(
            label_rect.left() + 4.0 + state.depth as f32 * 18.0,
            label_rect.center().y - 6.0,
        ),
        vec2(12.0, 12.0),
    );
    let icon_response = if state.expandable {
        Some(ui.interact(
            icon_rect,
            ui.make_persistent_id(("row-expand", state.key)),
            Sense::click(),
        ))
    } else {
        None
    };
    if state.expandable {
        draw_disclosure_icon(ui.painter(), icon_rect, state.expanded);
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
        &ctx.store.changes[signal.id],
        state.key.bit_offset,
        ty,
        state.display_options,
        viewport.pixels_per_time,
        viewport.time_view_start,
        viewport.max_time,
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
        .map(|pos| {
            time_from_pointer(
                wave_rect,
                pos,
                viewport.pixels_per_time,
                viewport.time_view_start,
                viewport.max_time,
            )
        });
    let secondary_cursor_time = wave_response
        .interact_pointer_pos()
        .filter(|pos| alt_down && wave_response.clicked() && wave_rect.contains(*pos))
        .map(|pos| {
            time_from_pointer(
                wave_rect,
                pos,
                viewport.pixels_per_time,
                viewport.time_view_start,
                viewport.max_time,
            )
        });
    let cursor_drag_started = !alt_down && !ui.input(|input| input.modifiers.shift) && wave_response.drag_started();
    let expand_toggles = icon_response
        .filter(|response| response.clicked())
        .map(|_| vec![state.key])
        .unwrap_or_default();

    RowResult {
        clicked: label_response.clicked() || wave_response.clicked(),
        label_drag_started: state.can_reorder && label_response.drag_started(),
        cursor_time,
        secondary_cursor_time,
        cursor_drag_started,
        expand_toggles,
        clicked_key: (!state.can_reorder && (label_response.clicked() || wave_response.clicked())).then_some(state.key),
        context_key: (label_response.secondary_clicked() || wave_response.secondary_clicked()).then_some(state.key),
        rect: row_rect,
        label_rect,
    }
}

pub fn draw_insert_line_for_targets(ui: &Ui, target_index: usize, targets: &[DropTarget], width: f32, fallback_y: f32) {
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
