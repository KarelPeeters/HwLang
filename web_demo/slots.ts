// Save slots of the demo, shown as tabs and kept in local storage.
// Every change re-reads the stored slots first, so multiple open demo pages don't drop each other's slots.
// If local storage is not available (eg. blocked by the browser), the slots only live in memory.

export interface Slot {
    id: string;
    source: string;
}

export interface SlotState {
    slots: Slot[];
    active: string;
}

export const STORAGE_KEY = "khdl-demo-slots";

function is_slot(value: any): value is Slot {
    return typeof value?.id === "string" && typeof value?.source === "string";
}

function read_storage(): SlotState | null {
    try {
        const value = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null");
        if (Array.isArray(value?.slots) && value.slots.length > 0 && value.slots.every(is_slot) && typeof value.active === "string") {
            return value;
        }
    } catch {
        // invalid or inaccessible storage, fall back to the in-memory state
    }
    return null;
}

function write_storage(state: SlotState) {
    try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
        // storage full or inaccessible, keep the in-memory state
    }
}

export function new_slot_id(): string {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

// Name shown on the tab: the first module in the source.
export function slot_title(source: string): string {
    const match = /\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)/.exec(source);
    return match ? match[1] : "untitled";
}

export class SlotStore {
    private state: SlotState;

    constructor(initial: () => SlotState) {
        this.state = read_storage() ?? initial();
        this.ensure_valid(this.state);
        write_storage(this.state);
    }

    get slots(): Slot[] {
        return this.state.slots;
    }

    get active(): Slot {
        return this.state.slots.find(slot => slot.id === this.state.active);
    }

    get(id: string): Slot | undefined {
        return this.state.slots.find(slot => slot.id === id);
    }

    // Pick up changes made by other demo pages.
    reload() {
        const stored = read_storage();
        if (stored !== null) {
            this.state = stored;
            this.ensure_valid(this.state);
        }
    }

    // Apply a change on top of the latest stored state, and store the result.
    update(change: (state: SlotState) => void) {
        this.reload();
        change(this.state);
        this.ensure_valid(this.state);
        write_storage(this.state);
    }

    set_source(id: string, source: string) {
        this.update(state => {
            const slot = state.slots.find(slot => slot.id === id);
            if (slot) {
                slot.source = source;
            } else {
                // removed by another page while we were editing it, keep the code
                state.slots.push({id, source});
            }
        });
    }

    set_active(id: string) {
        this.update(state => state.active = id);
    }

    // Open the given source in a new slot, or switch to an existing slot that already has exactly this source.
    open_source(source: string): Slot {
        let id: string = null;
        this.update(state => {
            const existing = state.slots.find(slot => slot.source === source);
            id = existing?.id ?? new_slot_id();
            if (!existing) {
                state.slots.push({id, source});
            }
            state.active = id;
        });
        return this.get(id);
    }

    remove(id: string, fallback_source: string) {
        this.update(state => {
            const index = state.slots.findIndex(slot => slot.id === id);
            if (index === -1) return;
            state.slots.splice(index, 1);
            if (state.slots.length === 0) {
                state.slots.push({id: new_slot_id(), source: fallback_source});
            }
            if (state.active === id) {
                state.active = state.slots[Math.min(index, state.slots.length - 1)].id;
            }
        });
    }

    private ensure_valid(state: SlotState) {
        if (!state.slots.some(slot => slot.id === state.active)) {
            state.active = state.slots[0].id;
        }
    }
}
