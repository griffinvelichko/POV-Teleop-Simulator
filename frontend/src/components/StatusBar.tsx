"use client";

import { useTeleopState } from "@/hooks/useTeleopState";

export default function StatusBar() {
  const { connected, fps } = useTeleopState();

  return (
    <div className="flex items-center justify-between px-4 py-2 bg-tron-panel/50 border-t border-tron-border">
      <div className="flex items-center gap-2">
        <div
          className={`w-2 h-2 rounded-full status-dot-pulse ${
            connected ? "bg-tron-cyan text-tron-cyan" : "bg-red-500 text-red-500"
          }`}
        />
        <span className="font-[family-name:var(--font-mono)] text-xs text-gray-400">
          {connected ? "ONLINE" : "OFFLINE"}
        </span>
      </div>
      <div className="flex items-center gap-4">
        <span className="font-[family-name:var(--font-mono)] text-xs text-gray-500">
          POV TELEOP v1.0
        </span>
        <span className="font-[family-name:var(--font-mono)] text-xs text-gray-400">
          {fps > 0 ? `${fps.toFixed(0)} FPS` : "--"}
        </span>
      </div>
    </div>
  );
}
