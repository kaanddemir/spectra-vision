import { useHealth } from "../api/queries";
import { Popover } from "../ui/Popover";
import { Spinner } from "../ui/Spinner";
import "./ServerStatus.css";

type Tone = "neutral" | "success" | "warning" | "danger";

const MEMORY: Record<string, [Tone, string]> = {
  normal: ["success", "Normal"],
  warning: ["warning", "Warning"],
  critical: ["danger", "Critical"],
};

/**
 * Compact header indicator of analysis readiness. "Ready" follows the server's own readiness
 * decision, so a memory warning that still permits analysis stays ready. The panel adds the
 * version, model and memory states without repeating the readiness label.
 */
export function ServerStatus() {
  const health = useHealth();
  const data = health.data;
  const offline = health.isError;

  const [tone, label]: [Tone, string] = health.isPending
    ? ["neutral", "Connecting"]
    : offline || !data
      ? ["danger", "Offline"]
      : data.analysis.available
        ? ["success", "Ready"]
        : ["warning", "Unavailable"];
  const accessibleName = offline ? "Server offline" : data ? `Analysis ${data.analysis.available ? "available" : "unavailable"}` : label;
  const [memoryTone, memoryLabel] = (data && MEMORY[data.memory.state]) ?? ["neutral", "Unknown"];

  return (
    <Popover
      panelClassName="server-status__panel"
      trigger={(props) => (
        <button type="button" className={`header-control server-status--${tone}`} aria-label={`${accessibleName}. Server details`} {...props}>
          {health.isPending ? <Spinner /> : <span className="server-status__dot" aria-hidden="true" />}
          <span className="server-status__label" aria-hidden="true">{label}</span>
        </button>
      )}
    >
      <p className="server-status__title">Local Server</p>
      {offline || !data ? (
        <p className="server-status__note">{offline ? "Not responding." : "Checking…"}</p>
      ) : (
        <>
          {data.analysis.reasons.length > 0 ? (
            <ul className="server-status__reasons">
              {data.analysis.reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          ) : null}
          <dl className="server-status__rows">
            <Row term="Version" value={data.version} />
            <Row term="Models" value={data.models.available ? "Ready" : "Unavailable"} tone={data.models.available ? "success" : "danger"} />
            <Row term="Memory" value={memoryLabel} tone={memoryTone} />
          </dl>
        </>
      )}
    </Popover>
  );
}

function Row({ term, value, tone }: { term: string; value: string; tone?: Tone }) {
  return (
    <div className="server-status__row">
      <dt>{term}</dt>
      <dd className={tone ? `server-status--${tone}` : undefined}>
        {tone ? <span className="server-status__dot" aria-hidden="true" /> : null}
        {value}
      </dd>
    </div>
  );
}
