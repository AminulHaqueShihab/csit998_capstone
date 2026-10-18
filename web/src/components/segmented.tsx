"use client";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

// Single-choice toggle (always one option selected).
export function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <ToggleGroup
      aria-label={label}
      variant="outline"
      value={[value]}
      onValueChange={(v) => {
        if (v.length) onChange(v[v.length - 1] as T);
      }}
      className="flex-wrap"
    >
      {options.map((o) => (
        <ToggleGroupItem key={o.value} value={o.value} className="px-3 text-xs aria-pressed:bg-primary! aria-pressed:text-primary-foreground! data-[state=on]:bg-primary! data-[state=on]:text-primary-foreground!">
          {o.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
