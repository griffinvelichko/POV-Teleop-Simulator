import StreamPanel from "@/components/StreamPanel";
import Dashboard from "@/components/Dashboard";
import StatusBar from "@/components/StatusBar";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-[#0a0a0f]">
      {/* Header */}
      <header className="flex items-center justify-center py-4 border-b border-tron-border">
        <h1 className="font-[family-name:var(--font-orbitron)] text-xl font-bold tracking-[0.3em] uppercase text-neon-cyan">
          POV Teleop
        </h1>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center p-4 gap-4 max-w-[1400px] mx-auto w-full">
        {/* Video Panels — client component handles direct backend URLs */}
        <StreamPanel />

        {/* Dashboard */}
        <div className="w-full">
          <Dashboard />
        </div>
      </main>

      {/* Status Bar */}
      <StatusBar />
    </div>
  );
}
