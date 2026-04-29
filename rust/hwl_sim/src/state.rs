use crate::render::WaveDisplayOptions;
use crate::rows::{RowDrag, WaveRow, WaveRowKey};
use crate::time::ZoomDrag;
use eframe::egui;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct SelectionState<T> {
    pub selected: BTreeSet<T>,
    pub primary: Option<T>,
    pub anchor: Option<T>,
}

impl<T> Default for SelectionState<T> {
    fn default() -> Self {
        Self {
            selected: BTreeSet::new(),
            primary: None,
            anchor: None,
        }
    }
}

impl<T: Ord + Clone> SelectionState<T> {
    pub fn clear(&mut self) {
        self.selected.clear();
        self.primary = None;
        self.anchor = None;
    }

    pub fn select_only(&mut self, item: T) {
        self.selected.clear();
        self.selected.insert(item.clone());
        self.primary = Some(item.clone());
        self.anchor = Some(item);
    }

    pub fn replace_selected(&mut self, selected: BTreeSet<T>, primary: Option<T>) {
        self.selected = selected;
        self.primary = primary.clone();
        self.anchor = primary;
    }

    pub fn apply_visible_selection(
        &mut self,
        item: T,
        visible_items: &[T],
        modifiers: egui::Modifiers,
        anchor_on_shift: bool,
    ) {
        if modifiers.shift {
            if let Some(anchor) = self.anchor.clone() {
                let start = visible_items.iter().position(|candidate| candidate == &anchor);
                let end = visible_items.iter().position(|candidate| candidate == &item);
                if let (Some(start), Some(end)) = (start, end) {
                    let (start, end) = if start <= end { (start, end) } else { (end, start) };
                    self.selected.extend(visible_items[start..=end].iter().cloned());
                } else {
                    self.selected.insert(item.clone());
                }
            } else {
                self.selected.insert(item.clone());
            }
            if anchor_on_shift {
                self.anchor = Some(item.clone());
            }
        } else if modifiers.ctrl || modifiers.command {
            if !self.selected.remove(&item) {
                self.selected.insert(item.clone());
            }
            self.anchor = Some(item.clone());
        } else {
            self.selected.clear();
            self.selected.insert(item.clone());
            self.anchor = Some(item.clone());
        }
        self.primary = Some(item);
    }

    pub fn apply_single_selection(&mut self, item: T, modifiers: egui::Modifiers) {
        if modifiers.ctrl || modifiers.command {
            if !self.selected.remove(&item) {
                self.selected.insert(item.clone());
            }
            self.primary = Some(item.clone());
            self.anchor = Some(item);
        } else {
            self.select_only(item);
        }
    }

    pub fn preserve_or_select(&mut self, item: T) {
        if !self.selected.contains(&item) {
            self.selected.clear();
            self.selected.insert(item.clone());
        }
        self.primary = Some(item.clone());
        self.anchor = Some(item);
    }
}

#[derive(Debug, Clone, Default)]
pub struct CursorState {
    pub primary: u64,
    pub secondary: Option<u64>,
    pub pending_secondary: Option<u64>,
    pub dragging_primary: bool,
}

impl CursorState {
    pub fn clear_secondary(&mut self) {
        self.secondary = None;
        self.pending_secondary = None;
    }

    pub fn update_pending_secondary(&mut self, time: u64) {
        self.pending_secondary = Some(time);
        self.secondary = Some(time);
    }

    pub fn commit_pending_secondary(&mut self) {
        if let Some(time) = self.pending_secondary.take() {
            self.secondary = Some(time);
        }
    }

    pub fn set_secondary(&mut self, time: u64) {
        self.secondary = Some(time);
    }
}

#[derive(Debug, Clone, Default)]
pub enum DragState {
    #[default]
    None,
    Row(RowDrag),
    Zoom(ZoomDrag),
    Signals(Vec<usize>),
}

impl DragState {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub fn row(&self) -> Option<&RowDrag> {
        match self {
            Self::Row(row) => Some(row),
            Self::None | Self::Zoom(_) | Self::Signals(_) => None,
        }
    }

    pub fn row_mut(&mut self) -> Option<&mut RowDrag> {
        match self {
            Self::Row(row) => Some(row),
            Self::None | Self::Zoom(_) | Self::Signals(_) => None,
        }
    }

    pub fn take_row(&mut self) -> Option<RowDrag> {
        match std::mem::take(self) {
            Self::Row(row) => Some(row),
            other => {
                *self = other;
                None
            }
        }
    }

    pub fn zoom(&self) -> Option<ZoomDrag> {
        match self {
            Self::Zoom(zoom) => Some(*zoom),
            Self::None | Self::Row(_) | Self::Signals(_) => None,
        }
    }

    pub fn zoom_mut(&mut self) -> Option<&mut ZoomDrag> {
        match self {
            Self::Zoom(zoom) => Some(zoom),
            Self::None | Self::Row(_) | Self::Signals(_) => None,
        }
    }

    pub fn take_zoom(&mut self) -> Option<ZoomDrag> {
        match std::mem::take(self) {
            Self::Zoom(zoom) => Some(zoom),
            other => {
                *self = other;
                None
            }
        }
    }

    pub fn signal_ids(&self) -> &[usize] {
        match self {
            Self::Signals(signals) => signals,
            Self::None | Self::Row(_) | Self::Zoom(_) => &[],
        }
    }

    pub fn clear_signals_if_idle(&mut self, primary_down: bool) {
        if !primary_down && matches!(self, Self::Signals(_)) {
            *self = Self::None;
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RowKeyedState {
    pub expanded: BTreeSet<WaveRowKey>,
    pub selected_subsections: SelectionState<WaveRowKey>,
    pub display_options: BTreeMap<WaveRowKey, WaveDisplayOptions>,
}

impl RowKeyedState {
    pub fn retain_existing_rows(&mut self, rows: &[WaveRow]) {
        let row_ids = rows.iter().map(|row| row.id).collect::<BTreeSet<_>>();
        self.expanded.retain(|key| row_ids.contains(&key.row_id));
        self.selected_subsections
            .selected
            .retain(|key| row_ids.contains(&key.row_id));
        self.display_options.retain(|key, _| row_ids.contains(&key.row_id));
    }

    pub fn toggle_expanded(&mut self, key: WaveRowKey) {
        if self.expanded.remove(&key) {
            self.selected_subsections
                .selected
                .retain(|selected_key| selected_key.row_id != key.row_id);
        } else {
            self.expanded.insert(key);
        }
    }
}
