use crate::consts::{
    COLOR_ADDED_SIGNAL_BG, COLOR_ROW_HOVER_BG, COLOR_ROW_SELECTED_BG, COLOR_TEXT_MUTED, COLOR_TEXT_PRIMARY,
    COLOR_TRANSPARENT,
};
use crate::rows::WaveRow;
use crate::state::SelectionState;
use crate::widgets::draw_disclosure_icon;
use eframe::egui::{Align2, FontId, Rect, Sense, Ui, pos2, vec2};
use hwl_language::sim::recorder::{WaveSignalKind, WaveStore};
use std::collections::BTreeSet;

struct TreeHeaderResult {
    clicked: bool,
    expanded: bool,
}

pub fn draw_hierarchy(
    ui: &mut Ui,
    store: &WaveStore,
    module_selection: &mut SelectionState<Vec<String>>,
    signal_selection: &mut SelectionState<usize>,
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
            &module_selection.selected,
            collapsed_hierarchy,
            has_submodules,
            0,
        );
        if header.clicked {
            update_module_selection(ui, store, &next_prefix, module_selection);
            signal_selection.clear();
        }
        if header.expanded {
            ui.indent(key, |ui| {
                draw_hierarchy(
                    ui,
                    store,
                    module_selection,
                    signal_selection,
                    collapsed_hierarchy,
                    &next_prefix,
                )
            });
        }
    }
}

pub fn draw_signal_panel_row(
    ui: &mut Ui,
    store: &WaveStore,
    visible_signals: &[usize],
    signal_id: usize,
    rows: &[WaveRow],
    signal_selection: &mut SelectionState<usize>,
) -> Option<Vec<usize>> {
    let signal = &store.signals[signal_id];
    let already_added = rows.iter().any(|row| row.signal_id() == Some(signal.id));
    let selected = signal_selection.selected.contains(&signal.id);
    let row_height = 22.0;
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), row_height), Sense::click_and_drag());
    if selected {
        ui.painter().rect_filled(rect, 2.0, COLOR_ROW_SELECTED_BG);
    } else if already_added {
        ui.painter().rect_filled(rect, 2.0, COLOR_ADDED_SIGNAL_BG);
    } else if response.hovered() {
        ui.painter().rect_filled(rect, 2.0, COLOR_ROW_HOVER_BG);
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
        COLOR_TEXT_MUTED,
    );
    ui.painter().text(
        pos2(rect.left() + 64.0, rect.center().y),
        Align2::LEFT_CENTER,
        format!("{}.{}", signal.path.join("."), signal.name),
        FontId::proportional(13.0),
        COLOR_TEXT_PRIMARY,
    );
    if response.clicked() {
        update_signal_selection(ui, visible_signals, signal.id, signal_selection);
    }
    if response.drag_started() || response.dragged() {
        if !signal_selection.selected.contains(&signal.id) {
            signal_selection.select_only(signal.id);
        }
        return Some(signal_selection.selected.iter().copied().collect());
    }
    None
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
    let icon_rect = Rect::from_min_size(
        pos2(rect.left() + 4.0 + indent, rect.center().y - 6.0),
        vec2(12.0, 12.0),
    );
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
        COLOR_ROW_SELECTED_BG
    } else if label_response.hovered() || icon_response.as_ref().is_some_and(|response| response.hovered()) {
        COLOR_ROW_HOVER_BG
    } else {
        COLOR_TRANSPARENT
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
        COLOR_TEXT_PRIMARY,
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

pub fn filtered_signal_ids(
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

pub fn filtered_signal_ids_for_module(
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

fn update_signal_selection(
    ui: &Ui,
    visible_signals: &[usize],
    signal_id: usize,
    selection: &mut SelectionState<usize>,
) {
    let modifiers = ui.input(|input| input.modifiers);
    selection.apply_visible_selection(signal_id, visible_signals, modifiers, false);
}

fn update_module_selection(ui: &Ui, store: &WaveStore, path: &[String], selection: &mut SelectionState<Vec<String>>) {
    let modifiers = ui.input(|input| input.modifiers);
    let path = path.to_owned();
    let paths = module_paths(store);
    selection.apply_visible_selection(path, &paths, modifiers, true);
}
