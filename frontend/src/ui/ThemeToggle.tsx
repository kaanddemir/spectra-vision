import { useTheme } from "../theme/ThemeProvider";
import type { ThemePreference } from "../theme/theme";
import { Icon } from "./Icon";
import { icons } from "./icons";
import { Popover } from "./Popover";
import "./ThemeToggle.css";

const OPTIONS: { value: ThemePreference; label: string; icon: typeof icons.themeSystem }[] = [
  { value: "system", label: "System", icon: icons.themeSystem },
  { value: "light", label: "Light", icon: icons.themeLight },
  { value: "dark", label: "Dark", icon: icons.themeDark },
];

/** One header button opening native radio buttons: arrow keys move between them and the choice is announced. */
export function ThemeToggle() {
  const { preference, setPreference } = useTheme();
  const current = OPTIONS.find((option) => option.value === preference) ?? OPTIONS[0]!;
  return (
    <Popover
      panelClassName="theme-toggle__panel"
      trigger={(props) => (
        <button type="button" className="header-control" aria-label={`Colour theme: ${current.label}`} {...props}>
          <Icon icon={current.icon} />
        </button>
      )}
    >
      {(close) => (
        <fieldset className="theme-toggle">
          <legend className="visually-hidden">Colour theme</legend>
          {OPTIONS.map((option) => (
            <label key={option.value} className="theme-toggle__option">
              <input
                type="radio"
                name="theme"
                value={option.value}
                checked={preference === option.value}
                data-autofocus={preference === option.value || undefined}
                onChange={() => setPreference(option.value)}
                onClick={(event) => event.detail > 0 && close()}
                className="visually-hidden"
              />
              <Icon icon={option.icon} />
              <span className="theme-toggle__label">{option.label}</span>
              {preference === option.value ? <Icon icon={icons.check} className="theme-toggle__check" /> : null}
            </label>
          ))}
        </fieldset>
      )}
    </Popover>
  );
}
