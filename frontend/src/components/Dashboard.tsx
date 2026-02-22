"use client";

import { useTeleopState } from "@/hooks/useTeleopState";

const JOINT_NAMES = ["pan", "lift", "elbow", "wrist", "roll", "grip"];

function JointRow({
  label,
  values,
  color,
}: {
  label: string;
  values: number[];
  color: "cyan" | "orange";
}) {
  const textClass = color === "cyan" ? "text-neon-cyan" : "text-neon-orange";

  return (
    <div className="flex items-center gap-3">
      <span
        className={`font-[family-name:var(--font-orbitron)] text-xs font-bold w-6 shrink-0 ${textClass}`}
      >
        {label}
      </span>
      <div className="flex gap-3 flex-wrap">
        {JOINT_NAMES.map((name, i) => (
          <div key={name} className="flex items-baseline gap-1">
            <span className="text-xs text-gray-500">{name}</span>
            <span className="text-sm font-[family-name:var(--font-mono)] text-gray-200 tabular-nums w-14 text-right">
              {values[i] !== undefined ? (values[i] >= 0 ? "+" : "") + values[i].toFixed(2) : "--"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const state = useTeleopState();

  return (
    <div className="px-4 py-3 space-y-2 bg-tron-panel border border-tron-border rounded-lg">
      <div className="flex items-center justify-between mb-2">
        <span className="font-[family-name:var(--font-orbitron)] text-xs font-semibold tracking-[0.2em] uppercase text-gray-500">
          Joint Telemetry
        </span>
        <span className="font-[family-name:var(--font-mono)] text-sm text-green-400">
          {state.fps > 0 ? `${state.fps.toFixed(0)} FPS` : "-- FPS"}
        </span>
      </div>
      <JointRow label="R" values={state.right_action} color="cyan" />
      <JointRow label="L" values={state.left_action} color="orange" />
    </div>
  );
}
