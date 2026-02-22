"use client";

import { useEffect, useRef, useSyncExternalStore } from "react";

interface TeleopState {
  right_action: number[];
  left_action: number[];
  fps: number;
  connected: boolean;
}

const DEFAULT_STATE: TeleopState = {
  right_action: [],
  left_action: [],
  fps: 0,
  connected: false,
};

// Singleton store — single poll shared by all subscribers
let currentState: TeleopState = DEFAULT_STATE;
let listeners: Set<() => void> = new Set();
let polling = false;
let timeoutId: ReturnType<typeof setTimeout> | null = null;
let subscriberCount = 0;

function notify() {
  listeners.forEach((l) => l());
}

async function poll() {
  try {
    const res = await fetch("/api/state");
    if (res.ok) {
      currentState = await res.json();
    } else {
      currentState = { ...currentState, connected: false };
    }
  } catch {
    currentState = { ...DEFAULT_STATE };
  }
  notify();

  if (polling) {
    // Poll faster when connected (250ms), slower when not (2s)
    const delay = currentState.connected ? 250 : 2000;
    timeoutId = setTimeout(poll, delay);
  }
}

function startPolling() {
  if (polling) return;
  polling = true;
  poll();
}

function stopPolling() {
  polling = false;
  if (timeoutId) {
    clearTimeout(timeoutId);
    timeoutId = null;
  }
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  subscriberCount++;
  if (subscriberCount === 1) startPolling();
  return () => {
    listeners.delete(listener);
    subscriberCount--;
    if (subscriberCount === 0) stopPolling();
  };
}

function getSnapshot(): TeleopState {
  return currentState;
}

export function useTeleopState(): TeleopState {
  return useSyncExternalStore(subscribe, getSnapshot, () => DEFAULT_STATE);
}
