use crate::consts::{ROW_HEIGHT, TERMINAL_DROP_SLOT_SPACING};
use crate::state::SelectionState;
use eframe::egui::{self, Rect, Sense, Ui, pos2};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct RowDrag {
    pub row_ids: BTreeSet<u64>,
    pub first_id: u64,
    pub target_index: usize,
    pub placement: Option<DropPlacement>,
}

#[derive(Debug, Copy, Clone)]
pub struct DropPlacement {
    pub row_index: usize,
    pub parent: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct WaveRow {
    pub id: u64,
    pub kind: WaveRowKind,
}

#[derive(Debug, Clone)]
pub enum WaveRowKind {
    Signal {
        signal_id: usize,
        parent: Option<u64>,
    },
    Group {
        name: String,
        collapsed: bool,
        parent: Option<u64>,
        editing: bool,
    },
    Spacer {
        parent: Option<u64>,
    },
}

impl WaveRow {
    pub fn signal_id(&self) -> Option<usize> {
        match self.kind {
            WaveRowKind::Signal { signal_id, .. } => Some(signal_id),
            WaveRowKind::Group { .. } | WaveRowKind::Spacer { .. } => None,
        }
    }

    pub fn parent_id(&self) -> Option<u64> {
        match self.kind {
            WaveRowKind::Signal { parent, .. } => parent,
            WaveRowKind::Group { parent, .. } => parent,
            WaveRowKind::Spacer { parent } => parent,
        }
    }

    pub fn set_parent_id(&mut self, new_parent: Option<u64>) {
        match &mut self.kind {
            WaveRowKind::Signal { parent, .. } | WaveRowKind::Group { parent, .. } | WaveRowKind::Spacer { parent } => {
                *parent = new_parent
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WaveRowKey {
    pub row_id: u64,
    pub signal_id: usize,
    pub bit_offset: usize,
    pub bit_len: usize,
    pub part: u64,
}

#[derive(Clone)]
pub struct VisibleWaveRow {
    pub row_index: usize,
    pub row_id: u64,
    pub depth: usize,
    pub kind: VisibleWaveRowKind,
}

#[derive(Clone, Copy)]
pub struct DropTarget {
    pub row_index: usize,
    pub parent: Option<u64>,
    pub depth: usize,
    pub y: f32,
    pub rect: Rect,
}

#[derive(Clone)]
pub enum VisibleWaveRowKind {
    Signal { signal_id: usize },
    Group,
    Spacer,
}

pub fn visible_wave_rows(rows: &[WaveRow]) -> Vec<VisibleWaveRow> {
    let row_parents = row_parent_map(rows);
    let collapsed_groups = rows
        .iter()
        .filter_map(|row| match &row.kind {
            WaveRowKind::Group { collapsed: true, .. } => Some(row.id),
            _ => None,
        })
        .collect::<BTreeSet<_>>();

    rows.iter()
        .enumerate()
        .filter_map(|(row_index, row)| {
            if has_collapsed_ancestor(row.parent_id(), &row_parents, &collapsed_groups) {
                return None;
            }
            let depth = group_depth(row.parent_id(), &row_parents);
            match row.kind {
                WaveRowKind::Group { .. } => Some(VisibleWaveRow {
                    row_index,
                    row_id: row.id,
                    depth,
                    kind: VisibleWaveRowKind::Group,
                }),
                WaveRowKind::Spacer { .. } => Some(VisibleWaveRow {
                    row_index,
                    row_id: row.id,
                    depth,
                    kind: VisibleWaveRowKind::Spacer,
                }),
                WaveRowKind::Signal { signal_id, .. } => Some(VisibleWaveRow {
                    row_index,
                    row_id: row.id,
                    depth,
                    kind: VisibleWaveRowKind::Signal { signal_id },
                }),
            }
        })
        .collect()
}

pub fn row_parent_map(rows: &[WaveRow]) -> BTreeMap<u64, Option<u64>> {
    rows.iter().map(|row| (row.id, row.parent_id())).collect()
}

fn has_collapsed_ancestor(
    mut parent: Option<u64>,
    row_parents: &BTreeMap<u64, Option<u64>>,
    collapsed_groups: &BTreeSet<u64>,
) -> bool {
    while let Some(parent_id) = parent {
        if collapsed_groups.contains(&parent_id) {
            return true;
        }
        parent = row_parents.get(&parent_id).copied().flatten();
    }
    false
}

fn group_depth(mut parent: Option<u64>, row_parents: &BTreeMap<u64, Option<u64>>) -> usize {
    let mut depth = 0;
    while let Some(parent_id) = parent {
        depth += 1;
        parent = row_parents.get(&parent_id).copied().flatten();
    }
    depth
}

fn is_descendant_of(row_id: u64, ancestor_id: u64, row_parents: &BTreeMap<u64, Option<u64>>) -> bool {
    let mut parent = row_parents.get(&row_id).copied().flatten();
    while let Some(parent_id) = parent {
        if parent_id == ancestor_id {
            return true;
        }
        parent = row_parents.get(&parent_id).copied().flatten();
    }
    false
}

pub fn included_row_ids(
    rows: &[WaveRow],
    selected_rows: &BTreeSet<u64>,
    row_parents: &BTreeMap<u64, Option<u64>>,
) -> BTreeSet<u64> {
    rows.iter()
        .filter_map(|row| {
            (selected_rows.contains(&row.id)
                || selected_rows
                    .iter()
                    .any(|selected_id| is_descendant_of(row.id, *selected_id, row_parents)))
            .then_some(row.id)
        })
        .collect()
}

pub fn drop_targets_for_rows(
    rows: &[WaveRow],
    visible_rows: &[VisibleWaveRow],
    row_rects: &[Rect],
    excluded_ids: &BTreeSet<u64>,
) -> Vec<DropTarget> {
    let row_parents = row_parent_map(rows);
    let visible_by_row_index = visible_rows
        .iter()
        .enumerate()
        .map(|(visible_index, row)| (row.row_index, visible_index))
        .collect::<BTreeMap<_, _>>();
    let mut targets = Vec::new();

    push_child_slot_targets(
        rows,
        visible_rows,
        row_rects,
        &visible_by_row_index,
        &row_parents,
        excluded_ids,
        None,
        0,
        &mut targets,
    );

    for (row_index, row) in rows.iter().enumerate() {
        if excluded_ids.contains(&row.id) || !matches!(row.kind, WaveRowKind::Group { .. }) {
            continue;
        }
        let depth = group_depth(Some(row.id), &row_parents);
        if is_group_collapsed(row) {
            if let Some(visible_index) = visible_by_row_index.get(&row_index).copied() {
                let rect = row_rects[visible_index];
                targets.push(DropTarget {
                    row_index: row_block_end_index(rows, row_index, &row_parents),
                    parent: Some(row.id),
                    depth,
                    y: rect.bottom(),
                    rect,
                });
            }
        } else {
            push_child_slot_targets(
                rows,
                visible_rows,
                row_rects,
                &visible_by_row_index,
                &row_parents,
                excluded_ids,
                Some(row.id),
                depth,
                &mut targets,
            );
        }
    }

    separate_terminal_drop_targets(&mut targets, rows.len());
    targets.sort_by(|a, b| {
        a.y.total_cmp(&b.y)
            .then_with(|| b.depth.cmp(&a.depth))
            .then_with(|| a.row_index.cmp(&b.row_index))
    });
    targets
}

fn push_child_slot_targets(
    rows: &[WaveRow],
    visible_rows: &[VisibleWaveRow],
    row_rects: &[Rect],
    visible_by_row_index: &BTreeMap<usize, usize>,
    row_parents: &BTreeMap<u64, Option<u64>>,
    excluded_ids: &BTreeSet<u64>,
    parent: Option<u64>,
    depth: usize,
    targets: &mut Vec<DropTarget>,
) {
    let child_indices = rows
        .iter()
        .enumerate()
        .filter_map(|(row_index, row)| {
            (row.parent_id() == parent && !excluded_ids.contains(&row.id)).then_some(row_index)
        })
        .collect::<Vec<_>>();

    if child_indices.is_empty() {
        if let Some(parent_id) = parent {
            if let Some(parent_index) = rows.iter().position(|row| row.id == parent_id) {
                if let Some(visible_index) = visible_by_row_index.get(&parent_index).copied() {
                    let rect = row_rects[visible_index];
                    targets.push(DropTarget {
                        row_index: parent_index + 1,
                        parent,
                        depth,
                        y: rect.bottom(),
                        rect,
                    });
                }
            }
        }
        return;
    }

    if let Some(first_child_index) = child_indices.first().copied() {
        if let Some(rect) = visible_block_rect(
            rows,
            visible_rows,
            row_rects,
            visible_by_row_index,
            first_child_index,
            excluded_ids,
        ) {
            targets.push(DropTarget {
                row_index: first_child_index,
                parent,
                depth,
                y: rect.top(),
                rect,
            });
        }
    }

    for child_index in child_indices.iter().copied() {
        if let Some(rect) = visible_block_rect(
            rows,
            visible_rows,
            row_rects,
            visible_by_row_index,
            child_index,
            excluded_ids,
        ) {
            targets.push(DropTarget {
                row_index: row_block_end_index(rows, child_index, row_parents),
                parent,
                depth,
                y: rect.bottom(),
                rect,
            });
        }
    }
}

fn separate_terminal_drop_targets(targets: &mut [DropTarget], terminal_row_index: usize) {
    let Some(max_depth) = targets
        .iter()
        .filter_map(|target| (target.row_index == terminal_row_index).then_some(target.depth))
        .max()
    else {
        return;
    };

    for target in targets
        .iter_mut()
        .filter(|target| target.row_index == terminal_row_index)
    {
        let offset = (max_depth - target.depth) as f32 * TERMINAL_DROP_SLOT_SPACING;
        if offset == 0.0 {
            continue;
        }
        target.y += offset;
        target.rect = Rect::from_min_max(
            pos2(target.rect.left(), target.y - ROW_HEIGHT / 2.0),
            pos2(target.rect.right(), target.y + ROW_HEIGHT / 2.0),
        );
    }
}

fn visible_block_rect(
    rows: &[WaveRow],
    visible_rows: &[VisibleWaveRow],
    row_rects: &[Rect],
    visible_by_row_index: &BTreeMap<usize, usize>,
    row_index: usize,
    excluded_ids: &BTreeSet<u64>,
) -> Option<Rect> {
    let visible_index = visible_by_row_index.get(&row_index).copied()?;
    let depth = visible_rows[visible_index].depth;
    let mut rect = None;
    for (index, visible_row) in visible_rows.iter().enumerate().skip(visible_index) {
        if index > visible_index && visible_row.depth <= depth {
            break;
        }
        if !excluded_ids.contains(&visible_row.row_id) {
            rect = Some(rect.map_or(row_rects[index], |rect: Rect| rect.union(row_rects[index])));
        }
    }
    if rect.is_none() && !excluded_ids.contains(&rows[row_index].id) {
        rect = visible_by_row_index
            .get(&row_index)
            .copied()
            .map(|visible_index| row_rects[visible_index]);
    }
    rect
}

fn row_block_end_index(rows: &[WaveRow], row_index: usize, row_parents: &BTreeMap<u64, Option<u64>>) -> usize {
    let row_id = rows[row_index].id;
    let mut end = row_index + 1;
    while end < rows.len() && is_descendant_of(rows[end].id, row_id, row_parents) {
        end += 1;
    }
    end
}

fn is_group_collapsed(row: &WaveRow) -> bool {
    matches!(row.kind, WaveRowKind::Group { collapsed: true, .. })
}

pub fn drop_placement(target: &DropTarget) -> DropPlacement {
    DropPlacement {
        row_index: target.row_index,
        parent: target.parent,
    }
}

pub fn placement_after_row(rows: &[WaveRow], row_index: usize) -> DropPlacement {
    let row_parents = row_parent_map(rows);
    DropPlacement {
        row_index: row_block_end_index(rows, row_index, &row_parents),
        parent: rows.get(row_index).and_then(WaveRow::parent_id),
    }
}

pub fn selected_signal_row_ids<'a>(
    rows: &'a [WaveRow],
    selected_rows: &'a BTreeSet<u64>,
) -> impl Iterator<Item = u64> + 'a {
    rows.iter().filter_map(|row| {
        (selected_rows.contains(&row.id) && matches!(row.kind, WaveRowKind::Signal { .. })).then_some(row.id)
    })
}

pub fn drag_row_ids(rows: &[WaveRow], start_index: usize, selected_rows: &BTreeSet<u64>) -> BTreeSet<u64> {
    if start_index >= rows.len() {
        return BTreeSet::new();
    }
    if selected_rows.contains(&rows[start_index].id) && selected_rows.len() > 1 {
        let row_parents = row_parent_map(rows);
        return included_row_ids(rows, selected_rows, &row_parents);
    }
    match rows[start_index].kind {
        WaveRowKind::Group { .. } => {
            let group_id = rows[start_index].id;
            let row_parents = row_parent_map(rows);
            rows.iter()
                .filter_map(|row| {
                    (row.id == group_id || is_descendant_of(row.id, group_id, &row_parents)).then_some(row.id)
                })
                .collect()
        }
        WaveRowKind::Signal { .. } | WaveRowKind::Spacer { .. } => BTreeSet::from([rows[start_index].id]),
    }
}

pub fn drain_drag_rows_for_move(
    rows: &mut Vec<WaveRow>,
    included_ids: &BTreeSet<u64>,
    raw_insert_index: usize,
) -> (Vec<WaveRow>, usize) {
    let mut moved_rows = Vec::new();
    let mut kept_rows = Vec::new();
    let mut insert_index = 0;
    for (index, row) in rows.drain(..).enumerate() {
        let included = included_ids.contains(&row.id);
        if index < raw_insert_index && !included {
            insert_index += 1;
        }
        if included {
            moved_rows.push(row);
        } else {
            kept_rows.push(row);
        }
    }
    *rows = kept_rows;
    (moved_rows, insert_index.min(rows.len()))
}

pub fn reparent_drag_roots(rows: &mut [WaveRow], row_ids: &BTreeSet<u64>, new_parent: Option<u64>) {
    for row in rows {
        if row.parent_id().is_none_or(|parent| !row_ids.contains(&parent)) {
            row.set_parent_id(new_parent);
        }
    }
}

pub fn group_blocks_are_contiguous(rows: &[WaveRow]) -> bool {
    let row_parents = row_parent_map(rows);
    if rows
        .iter()
        .any(|row| row.parent_id().is_some_and(|parent| !row_parents.contains_key(&parent)))
    {
        return false;
    }

    for (group_index, group) in rows.iter().enumerate() {
        if !matches!(group.kind, WaveRowKind::Group { .. }) {
            continue;
        }
        if rows[..group_index]
            .iter()
            .any(|row| is_descendant_of(row.id, group.id, &row_parents))
        {
            return false;
        }

        let mut left_group_block = false;
        for row in &rows[group_index + 1..] {
            let descendant = is_descendant_of(row.id, group.id, &row_parents);
            if descendant && left_group_block {
                return false;
            }
            if !descendant {
                left_group_block = true;
            }
        }
    }
    true
}

pub fn delete_selected_rows(rows: &mut Vec<WaveRow>, selected_rows: &BTreeSet<u64>) {
    let row_parents = row_parent_map(rows);
    let included_ids = included_row_ids(rows, selected_rows, &row_parents);
    rows.retain(|row| !included_ids.contains(&row.id));
}

pub fn update_wave_row_selection(
    ui: &Ui,
    visible_index: usize,
    visible_rows: &[VisibleWaveRow],
    selection: &mut SelectionState<u64>,
) {
    let row_id = visible_rows[visible_index].row_id;
    let modifiers = ui.input(|input| input.modifiers);
    let visible_ids = visible_rows.iter().map(|row| row.row_id).collect::<Vec<_>>();
    selection.apply_visible_selection(row_id, &visible_ids, modifiers, false);
}

pub fn preserve_or_select_dragged_row(row_id: u64, selection: &mut SelectionState<u64>) {
    selection.preserve_or_select(row_id);
}

pub fn best_drop_target_index(pointer_pos: egui::Pos2, targets: &[DropTarget], label_width: f32) -> usize {
    targets
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            drop_target_score(pointer_pos, a, label_width)
                .total_cmp(&drop_target_score(pointer_pos, b, label_width))
                .then_with(|| drop_target_depth_tiebreak(pointer_pos, a, b, label_width))
        })
        .map(|(index, _)| index)
        .unwrap_or(targets.len())
}

fn drop_target_score(pointer_pos: egui::Pos2, target: &DropTarget, label_width: f32) -> f32 {
    let y_score = (pointer_pos.y - target.y).abs();
    let label_right = target.rect.left() + label_width;
    let x_score = if pointer_pos.x <= label_right {
        let target_x = target.rect.left() + 28.0 + target.depth as f32 * 18.0;
        (pointer_pos.x - target_x).abs().min(120.0) * 0.35
    } else {
        0.0
    };
    y_score + x_score
}

fn drop_target_depth_tiebreak(
    pointer_pos: egui::Pos2,
    a: &DropTarget,
    b: &DropTarget,
    label_width: f32,
) -> std::cmp::Ordering {
    let label_right = a.rect.left() + label_width;
    if pointer_pos.x <= label_right {
        b.depth.cmp(&a.depth)
    } else {
        a.depth.cmp(&b.depth)
    }
}

pub fn empty_panel_area_clicked(ui: &mut Ui) -> bool {
    let rect = ui.available_rect_before_wrap();
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return false;
    }
    ui.allocate_rect(rect, Sense::click())
        .on_hover_cursor(egui::CursorIcon::Default)
        .clicked()
}

#[cfg(test)]
mod tests {
    use crate::consts::{DEFAULT_ROW_LABEL_WIDTH, ROW_HEIGHT};

    use super::{
        DropTarget, VisibleWaveRow, WaveRow, WaveRowKind, best_drop_target_index, drain_drag_rows_for_move,
        drop_targets_for_rows, group_blocks_are_contiguous, reparent_drag_roots, visible_wave_rows,
    };
    use eframe::egui::{Rect, pos2, vec2};
    use std::collections::BTreeSet;

    fn signal(id: u64, parent: Option<u64>) -> WaveRow {
        WaveRow {
            id,
            kind: WaveRowKind::Signal {
                signal_id: id as usize,
                parent,
            },
        }
    }

    fn group(id: u64, parent: Option<u64>, collapsed: bool) -> WaveRow {
        WaveRow {
            id,
            kind: WaveRowKind::Group {
                name: format!("g{id}"),
                collapsed,
                parent,
                editing: false,
            },
        }
    }

    fn visible_rects(visible_rows: &[VisibleWaveRow]) -> Vec<Rect> {
        visible_rows
            .iter()
            .enumerate()
            .map(|(index, _)| Rect::from_min_size(pos2(0.0, index as f32 * ROW_HEIGHT), vec2(400.0, ROW_HEIGHT)))
            .collect()
    }

    fn placements(targets: &[DropTarget]) -> BTreeSet<(Option<u64>, usize)> {
        targets.iter().map(|target| (target.parent, target.row_index)).collect()
    }

    fn move_rows_to(rows: &mut Vec<WaveRow>, row_ids: BTreeSet<u64>, row_index: usize, parent: Option<u64>) {
        let (mut moved_rows, insert_index) = drain_drag_rows_for_move(rows, &row_ids, row_index);
        reparent_drag_roots(&mut moved_rows, &row_ids, parent);
        rows.splice(insert_index..insert_index, moved_rows);
    }

    #[test]
    fn drop_targets_cover_group_boundaries_and_empty_groups() {
        let rows = vec![
            group(1, None, false),
            signal(2, Some(1)),
            signal(3, Some(1)),
            group(4, None, false),
            signal(5, None),
        ];
        let visible_rows = visible_wave_rows(&rows);
        let rects = visible_rects(&visible_rows);
        let targets = drop_targets_for_rows(&rows, &visible_rows, &rects, &BTreeSet::new());
        let placements = placements(&targets);

        assert!(placements.contains(&(None, 0)), "can drop right before a group");
        assert!(placements.contains(&(Some(1), 1)), "can drop at the start of a group");
        assert!(
            placements.contains(&(Some(1), 2)),
            "can drop between rows inside a group"
        );
        assert!(placements.contains(&(Some(1), 3)), "can drop at the end of a group");
        assert!(placements.contains(&(None, 3)), "can drop right after a group");
        assert!(placements.contains(&(Some(4), 4)), "can drop into an empty group");
        assert!(placements.contains(&(None, 4)), "can drop after an empty group");
        assert!(placements.contains(&(None, 5)), "can drop at the end of the top level");
    }

    #[test]
    fn empty_group_target_uses_pointer_indent_to_choose_inside_or_after() {
        let rows = vec![group(1, None, false), signal(2, None)];
        let visible_rows = visible_wave_rows(&rows);
        let rects = visible_rects(&visible_rows);
        let targets = drop_targets_for_rows(&rows, &visible_rows, &rects, &BTreeSet::new());

        let inside_index = best_drop_target_index(pos2(50.0, ROW_HEIGHT), &targets, DEFAULT_ROW_LABEL_WIDTH);
        let outside_index = best_drop_target_index(pos2(25.0, ROW_HEIGHT), &targets, DEFAULT_ROW_LABEL_WIDTH);

        assert_eq!(targets[inside_index].parent, Some(1));
        assert_eq!(targets[inside_index].row_index, 1);
        assert_eq!(targets[outside_index].parent, None);
        assert_eq!(targets[outside_index].row_index, 1);
    }

    #[test]
    fn final_empty_group_has_reachable_inside_and_after_targets() {
        let rows = vec![group(1, None, false)];
        let visible_rows = visible_wave_rows(&rows);
        let rects = visible_rects(&visible_rows);
        let targets = drop_targets_for_rows(&rows, &visible_rows, &rects, &BTreeSet::new());

        let inside = targets
            .iter()
            .find(|target| target.parent == Some(1) && target.row_index == rows.len())
            .expect("missing drop target inside final empty group");
        let after = targets
            .iter()
            .find(|target| target.parent.is_none() && target.row_index == rows.len())
            .expect("missing drop target after final empty group");
        assert!(
            after.y > inside.y,
            "after-list target must be below inside-group target"
        );

        let inside_index = best_drop_target_index(pos2(50.0, inside.y), &targets, DEFAULT_ROW_LABEL_WIDTH);
        let after_index = best_drop_target_index(
            pos2(DEFAULT_ROW_LABEL_WIDTH + 80.0, after.y),
            &targets,
            DEFAULT_ROW_LABEL_WIDTH,
        );

        assert_eq!(targets[inside_index].parent, Some(1));
        assert_eq!(targets[after_index].parent, None);
    }

    #[test]
    fn wave_area_tiebreak_prefers_outside_when_boundary_is_shared() {
        let rows = vec![group(1, None, false), signal(2, None)];
        let visible_rows = visible_wave_rows(&rows);
        let rects = visible_rects(&visible_rows);
        let targets = drop_targets_for_rows(&rows, &visible_rows, &rects, &BTreeSet::new());
        let shared_boundary_y = rects[0].bottom();

        let target_index = best_drop_target_index(
            pos2(DEFAULT_ROW_LABEL_WIDTH + 80.0, shared_boundary_y),
            &targets,
            DEFAULT_ROW_LABEL_WIDTH,
        );

        assert_eq!(targets[target_index].parent, None);
        assert_eq!(targets[target_index].row_index, 1);
    }

    #[test]
    fn moving_row_into_group_reparents_only_drag_roots() {
        let mut rows = vec![group(1, None, false), signal(2, None), signal(3, Some(1))];
        let row_ids = BTreeSet::from([2]);
        let (mut moved_rows, insert_index) = drain_drag_rows_for_move(&mut rows, &row_ids, 1);
        reparent_drag_roots(&mut moved_rows, &row_ids, Some(1));
        rows.splice(insert_index..insert_index, moved_rows);

        assert_eq!(rows[1].parent_id(), Some(1));
        assert!(group_blocks_are_contiguous(&rows));
    }

    #[test]
    fn moving_rows_to_every_group_boundary_keeps_groups_contiguous() {
        for (row_index, parent, expected_order, expected_parent) in [
            (0, None, vec![5, 1, 2, 3, 4], None),
            (1, Some(1), vec![1, 5, 2, 3, 4], Some(1)),
            (2, Some(1), vec![1, 2, 5, 3, 4], Some(1)),
            (3, Some(1), vec![1, 2, 3, 5, 4], Some(1)),
            (3, None, vec![1, 2, 3, 5, 4], None),
            (4, Some(4), vec![1, 2, 3, 4, 5], Some(4)),
            (4, None, vec![1, 2, 3, 4, 5], None),
        ] {
            let mut rows = vec![
                group(1, None, false),
                signal(2, Some(1)),
                signal(3, Some(1)),
                group(4, None, false),
                signal(5, None),
            ];
            move_rows_to(&mut rows, BTreeSet::from([5]), row_index, parent);
            assert_eq!(
                rows.iter().map(|row| row.id).collect::<Vec<_>>(),
                expected_order,
                "unexpected order for target ({parent:?}, {row_index})"
            );
            assert_eq!(
                rows.iter().find(|row| row.id == 5).and_then(WaveRow::parent_id),
                expected_parent,
                "unexpected parent for target ({parent:?}, {row_index})"
            );
            assert!(
                group_blocks_are_contiguous(&rows),
                "group blocks split for target ({parent:?}, {row_index})"
            );
        }
    }
}
