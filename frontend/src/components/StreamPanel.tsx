"use client";

import MjpegStream from "./MjpegStream";
import NeonPanel from "./NeonPanel";

const BACKEND_URL = "http://localhost:8000";

export default function StreamPanel() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
      <NeonPanel label="Camera" color="cyan">
        <MjpegStream
          src={`${BACKEND_URL}/api/stream/camera`}
          alt="Camera feed"
        />
      </NeonPanel>
      <NeonPanel label="Simulator" color="orange">
        <MjpegStream
          src={`${BACKEND_URL}/api/stream/sim`}
          alt="Simulator view"
        />
      </NeonPanel>
    </div>
  );
}
