/**
 * Recorded observations of one analysed frame, drawn over the video. The SVG's view box is the recorded
 * frame size and it fills the same box as the <video> with the same letterboxing (`object-fit: contain`
 * and `xMidYMid meet`), so boxes keep their recorded source-pixel coordinates at any player size.
 */
import { useId } from "react";

import type { LaneBoundary, OverlayFrame } from "../../api/types";
import { CATEGORY_LABELS } from "./labels";
import "./OverlayDrawing.css";

export interface OverlayLayers {
  boxes: boolean;
  lane: boolean;
}

interface OverlayDrawingProps {
  frame: OverlayFrame;
  width: number;
  height: number;
  layers: OverlayLayers;
}

/** Points of a lane boundary in source pixels, near end first. */
export function boundaryPoints(boundary: LaneBoundary, width: number, height: number): string {
  const points = boundary.points.length > 0 ? boundary.points : [boundary.near, boundary.far];
  return points.map(([x, y]) => `${round(x * width)},${round(y * height)}`).join(" ");
}

const round = (value: number) => Math.round(value * 100) / 100;

export function OverlayDrawing({ frame, width, height, layers }: OverlayDrawingProps) {
  const id = useId();
  const fontSize = round(height * 0.032);
  const boundaries = layers.lane && frame.lane ? [frame.lane.left, frame.lane.right].filter((b) => b !== null) : [];
  return (
    <svg
      className="overlay-drawing"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
      data-testid="overlay-drawing"
    >
      {boundaries.map((boundary, index) => {
        const points = boundaryPoints(boundary, width, height);
        // Beyond evidence_y the boundary is the road model's extrapolation: drawn dashed, never as observed.
        const split = boundary.evidence_y === null ? null : round(boundary.evidence_y * height);
        const clip = `${id}-lane-${index}`;
        if (!boundary.measured || split === null) {
          return (
            <polyline
              key={index}
              className={`overlay-drawing__lane${boundary.measured ? "" : " overlay-drawing__lane--extrapolated"}`}
              points={points}
            />
          );
        }
        return (
          <g key={index}>
            <clipPath id={`${clip}-observed`}>
              <rect x={0} y={split} width={width} height={Math.max(0, height - split)} />
            </clipPath>
            <clipPath id={`${clip}-beyond`}>
              <rect x={0} y={0} width={width} height={Math.max(0, split)} />
            </clipPath>
            <polyline className="overlay-drawing__lane" points={points} clipPath={`url(#${clip}-observed)`} />
            <polyline
              className="overlay-drawing__lane overlay-drawing__lane--extrapolated"
              points={points}
              clipPath={`url(#${clip}-beyond)`}
            />
          </g>
        );
      })}
      {layers.boxes
        ? (frame.tracks ?? []).map((track) => (
            <g
              key={track.track_id}
              className={[
                "overlay-drawing__track",
                track.related ? "overlay-drawing__track--related" : "",
                track.predicted ? "overlay-drawing__track--predicted" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              data-track={track.track_id}
            >
              <rect
                x={track.box.x1}
                y={track.box.y1}
                width={round(track.box.x2 - track.box.x1)}
                height={round(track.box.y2 - track.box.y1)}
              />
              <text
                x={track.box.x1}
                y={track.box.y1 > fontSize * 1.4 ? round(track.box.y1 - fontSize * 0.4) : round(track.box.y2 + fontSize)}
                fontSize={fontSize}
                strokeWidth={round(fontSize * 0.25)}
              >
                {`#${track.track_id} ${CATEGORY_LABELS[track.category] ?? track.category}${track.predicted ? " (predicted)" : ""}`}
              </text>
            </g>
          ))
        : null}
    </svg>
  );
}
