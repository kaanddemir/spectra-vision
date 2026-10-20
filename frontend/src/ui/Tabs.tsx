import { useRef, type KeyboardEvent } from "react";

import "./Tabs.css";

export interface TabItem<T extends string> {
  id: T;
  label: string;
  count?: number;
}

interface TabsProps<T extends string> {
  label: string;
  items: TabItem<T>[];
  value: T;
  onChange: (value: T) => void;
  /** id of the element the tabs filter or switch. */
  controls: string;
}

/** Tabs with arrow-key navigation; counts are part of each tab's accessible name. */
export function Tabs<T extends string>({ label, items, value, onChange, controls }: TabsProps<T>) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const last = items.length - 1;
    const next =
      event.key === "ArrowRight" ? (index === last ? 0 : index + 1)
      : event.key === "ArrowLeft" ? (index === 0 ? last : index - 1)
      : event.key === "Home" ? 0
      : event.key === "End" ? last
      : null;
    if (next === null) return;
    event.preventDefault();
    onChange(items[next]!.id);
    refs.current[next]?.focus();
  }

  return (
    <div role="tablist" aria-label={label} className="tabs">
      {items.map((item, index) => {
        const selected = item.id === value;
        return (
          <button
            key={item.id}
            ref={(node) => {
              refs.current[index] = node;
            }}
            type="button"
            role="tab"
            id={`${controls}-tab-${item.id}`}
            aria-selected={selected}
            aria-controls={controls}
            tabIndex={selected ? 0 : -1}
            className="tabs__tab"
            onClick={() => onChange(item.id)}
            onKeyDown={(event) => onKeyDown(event, index)}
          >
            {item.label}
            {item.count !== undefined ? (
              <>
                {" "}
                <span className="tabs__count type-numeric">{item.count}</span>
              </>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
