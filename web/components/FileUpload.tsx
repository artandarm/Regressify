"use client";
import { useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  loading: boolean;
}

export function FileUpload({ onFile, loading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onFile(file);
  }

  return (
    <div
      onClick={() => !loading && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={[
        "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed",
        "px-8 py-16 text-center transition-colors select-none",
        loading
          ? "border-edge text-muted cursor-default"
          : dragging
          ? "border-accent bg-raised cursor-copy"
          : "border-edge hover:border-secondary cursor-pointer",
      ].join(" ")}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx"
        className="hidden"
        onChange={handleChange}
        disabled={loading}
      />
      {loading ? (
        <div className="flex items-center gap-2 text-secondary text-sm">
          <Spinner />
          Uploading…
        </div>
      ) : (
        <>
          <div className="text-3xl text-muted">↑</div>
          <p className="text-sm text-prose">Drop a CSV or Excel file here</p>
          <p className="text-xs text-muted">or click to browse</p>
        </>
      )}
    </div>
  );
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 100 16v-4l-3 3 3 3v-4a8 8 0 01-8-8z" />
    </svg>
  );
}

export function SpinnerFull({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-24 text-secondary">
      <svg className="animate-spin h-8 w-8" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
        <path className="opacity-80" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
      <p className="text-sm">{label}</p>
    </div>
  );
}
