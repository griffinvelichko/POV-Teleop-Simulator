"use client";

interface NeonPanelProps {
  label: string;
  color?: "cyan" | "orange";
  children: React.ReactNode;
  className?: string;
}

export default function NeonPanel({
  label,
  color = "cyan",
  children,
  className = "",
}: NeonPanelProps) {
  const borderClass = color === "cyan" ? "border-glow-cyan" : "border-glow-orange";
  const labelClass = color === "cyan" ? "text-neon-cyan" : "text-neon-orange";

  return (
    <div
      className={`relative border rounded-lg bg-tron-panel overflow-hidden scanlines ${borderClass} ${className}`}
    >
      <div className="absolute top-0 left-0 right-0 z-20 flex items-center px-3 py-1.5 bg-tron-panel/80 backdrop-blur-sm border-b border-inherit">
        <span
          className={`font-[family-name:var(--font-orbitron)] text-xs font-semibold tracking-[0.2em] uppercase ${labelClass}`}
        >
          {label}
        </span>
      </div>
      <div className="pt-8">{children}</div>
    </div>
  );
}
