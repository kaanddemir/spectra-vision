import type { LucideIcon } from "lucide-react";
import {
  forwardRef,
  useEffect,
  useId,
  useRef,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

import { Icon } from "./Icon";
import { icons } from "./icons";
import "./Field.css";

interface FieldFrame {
  label: string;
  /** Keeps the label for assistive technology only, for self-explanatory controls like search. */
  hideLabel?: boolean;
  description?: ReactNode;
  error?: string;
}

function Frame({ id, label, hideLabel, description, error, children }: FieldFrame & { id: string; children: ReactNode }) {
  return (
    <div className="field">
      <label htmlFor={id} className={hideLabel ? "visually-hidden" : "field__label"}>
        {label}
      </label>
      {children}
      {description ? (
        <p id={`${id}-description`} className="field__description">
          {description}
        </p>
      ) : null}
      {error ? (
        <p id={`${id}-error`} className="field__error">
          <Icon icon={icons.danger} />
          {error}
        </p>
      ) : null}
    </div>
  );
}

function describedBy(id: string, frame: FieldFrame): string | undefined {
  const ids = [frame.description ? `${id}-description` : null, frame.error ? `${id}-error` : null];
  return ids.filter(Boolean).join(" ") || undefined;
}

interface TextFieldProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "size">, FieldFrame {
  icon?: LucideIcon;
}

export const TextField = forwardRef<HTMLInputElement, TextFieldProps>(function TextField(
  { label, hideLabel, description, error, icon, className, id: givenId, ...rest },
  ref,
) {
  const generated = useId();
  const id = givenId ?? generated;
  const frame = { label, hideLabel, description, error };
  return (
    <Frame id={id} {...frame}>
      <div className={["control", icon ? "control--with-icon" : null, className].filter(Boolean).join(" ")}>
        {icon ? <Icon icon={icon} className="control__icon" /> : null}
        <input
          ref={ref}
          id={id}
          className="control__input"
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy(id, frame)}
          {...rest}
        />
      </div>
    </Frame>
  );
});

type TextAreaProps = TextareaHTMLAttributes<HTMLTextAreaElement> & FieldFrame;

/** Multi-line plain text with the same frame as `TextField`. */
export function TextArea({ label, hideLabel, description, error, className, id: givenId, ...rest }: TextAreaProps) {
  const generated = useId();
  const id = givenId ?? generated;
  const frame = { label, hideLabel, description, error };
  return (
    <Frame id={id} {...frame}>
      <div className={["control", "control--textarea", className].filter(Boolean).join(" ")}>
        <textarea
          id={id}
          className="control__input"
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy(id, frame)}
          {...rest}
        />
      </div>
    </Frame>
  );
}

type SearchFieldProps = Omit<TextFieldProps, "icon" | "type" | "hideLabel">;

/** A labelled search input; the label is visually hidden because the icon and placeholder explain it. */
export function SearchField(props: SearchFieldProps) {
  return <TextField type="search" icon={icons.search} hideLabel autoComplete="off" spellCheck={false} {...props} />;
}

interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, "size">, FieldFrame {
  options: { value: string; label: string }[];
}

/** A native select: accessible and keyboard-friendly on every platform. */
export function Select({ label, hideLabel, description, error, options, className, id: givenId, ...rest }: SelectProps) {
  const generated = useId();
  const id = givenId ?? generated;
  const frame = { label, hideLabel, description, error };
  return (
    <Frame id={id} {...frame}>
      <div className={["control", "control--select", className].filter(Boolean).join(" ")}>
        <select
          id={id}
          className="control__input"
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy(id, frame)}
          {...rest}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <Icon icon={icons.chevronDown} className="control__chevron" />
      </div>
    </Frame>
  );
}

interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "type" | "size"> {
  label: string;
  hideLabel?: boolean;
  indeterminate?: boolean;
}

export function Checkbox({ label, hideLabel = false, indeterminate = false, className, ...rest }: CheckboxProps) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = indeterminate;
  }, [indeterminate]);
  return (
    <label className={["checkbox", className].filter(Boolean).join(" ")}>
      <input ref={ref} type="checkbox" className="checkbox__input" {...rest} />
      <span className={hideLabel ? "visually-hidden" : "checkbox__label"}>{label}</span>
    </label>
  );
}
