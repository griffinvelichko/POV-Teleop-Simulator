"use client";

import { useRef, useEffect, useState, useCallback } from "react";

interface MjpegStreamProps {
  src: string;
  alt?: string;
  className?: string;
}

export default function MjpegStream({
  src,
  alt = "Video stream",
  className = "",
}: MjpegStreamProps) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [error, setError] = useState(false);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout>>(null);
  const retryCountRef = useRef(0);

  const connect = useCallback(() => {
    if (imgRef.current) {
      setError(false);
      imgRef.current.src = src;
    }
  }, [src]);

  useEffect(() => {
    connect();
    return () => {
      if (imgRef.current) {
        imgRef.current.src = "";
      }
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, [connect]);

  const handleError = () => {
    setError(true);
    // Exponential backoff: 1s, 2s, 4s, max 8s
    const delay = Math.min(1000 * Math.pow(2, retryCountRef.current), 8000);
    retryCountRef.current += 1;
    retryTimeoutRef.current = setTimeout(() => {
      connect();
    }, delay);
  };

  const handleLoad = () => {
    retryCountRef.current = 0;
    setError(false);
  };

  return (
    <div className={`relative w-full aspect-[4/3] bg-black ${className}`}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        ref={imgRef}
        alt={alt}
        onError={handleError}
        onLoad={handleLoad}
        className="w-full h-full object-contain"
      />
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80">
          <span className="font-[family-name:var(--font-mono)] text-sm text-tron-cyan-dim animate-pulse">
            Connecting...
          </span>
        </div>
      )}
    </div>
  );
}
