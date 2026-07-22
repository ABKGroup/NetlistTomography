#!/usr/bin/env python3
"""
SDC Clock Period Adjustment Script

Adjusts clock periods for bp_clk, bp_clock, core_clock, or clk in SDC
files.
Automatically detects units (ps if period > 50, ns if period <= 50).

Usage:
    python adjust_sdc_clock.py <input_sdc> <adjustment>

Arguments:
    input_sdc   : Path to input SDC file
    adjustment  : 0, 1, 2, -1, or -2
                  0  -> Copy file without changes
                  1  -> +0.1ps (+0.0001ns)
                  2  -> +0.2ps (+0.0002ns)
                  -1 -> -0.1ps (-0.0001ns)
                  -2 -> -0.2ps (-0.0002ns)

Output:
    <input_basename>_updated.sdc
"""

import sys
import re
import os


def adjust_sdc_clock(input_file, adjustment_param):
    """
    Adjust clock periods in SDC file.

    Args:
        input_file: Path to input SDC file
        adjustment_param: Integer (0, 1, 2, -1, -2)
    """
    # Validate adjustment parameter
    valid_params = [0, 1, 2, -1, -2]
    if adjustment_param not in valid_params:
        raise ValueError(
            f"Invalid adjustment parameter: {adjustment_param}. "
            f"Must be one of {valid_params}"
        )

    # Generate output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_updated.sdc"

    # Read input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # If adjustment is 0, just copy the file
    if adjustment_param == 0:
        with open(output_file, 'w') as f:
            f.writelines(lines)
        print(f"Copied {input_file} to {output_file} (no modifications)")
        return

    # Define adjustment mapping (in picoseconds)
    # Will be converted to ns if needed
    adjustment_ps = {
        1: 0.1,
        2: 0.2,
        -1: -0.1,
        -2: -0.2
    }

    adjustment_value_ps = adjustment_ps[adjustment_param]

    # Target clock names
    target_clocks = {'bp_clk', 'bp_clock', 'core_clock', 'clk'}

    # Pattern 1: [get_ports ...] -name ... -period ... -waveform ...
    # Groups: (port_spec) (clock_name) (period) (waveform_start)
    # (waveform_half)
    clock_pattern1 = re.compile(
        r'(create_clock\s+\[get_ports\s+\{[^}]+\}\]\s+'
        r'-name\s+)(\S+)(\s+-period\s+)(\S+)(\s+-waveform\s+\{)(\S+)'
        r'(\s+)(\S+)(\})'
    )

    # Pattern 2: -name ... -period ... -waveform ... [get_ports ...]
    # Groups: (prefix) (clock_name) (period) (waveform_start)
    # (waveform_half) (port_spec)
    clock_pattern2 = re.compile(
        r'(create_clock\s+-name\s+)(\S+)(\s+-period\s+)(\S+)'
        r'(\s+-waveform\s+\{)(\S+)(\s+)(\S+)(\}\s+\[get_ports\s+)'
        r'([^\]]+)(\])'
    )

    modified_lines = []
    modifications_count = 0

    for line in lines:
        # Try pattern 1 first
        match1 = clock_pattern1.search(line)
        # Try pattern 2 if pattern 1 doesn't match
        match2 = clock_pattern2.search(line) if not match1 else None

        match = match1 or match2
        pattern_type = 1 if match1 else (2 if match2 else 0)

        if match:
            clock_name = match.group(2)
            # Strip quotes if present for comparison
            clock_name_stripped = clock_name.strip('"')

            # Only modify target clocks
            if clock_name_stripped in target_clocks:
                period_str = match.group(4)
                period = float(period_str)

                # Detect units based on period value
                is_ps = period > 50

                # Apply adjustment
                if is_ps:
                    # Period is in ps, add adjustment directly
                    new_period = period + adjustment_value_ps
                else:
                    # Period is in ns, convert adjustment from ps to ns
                    adjustment_ns = adjustment_value_ps / 1000.0
                    new_period = period + adjustment_ns

                # Calculate new waveform (50% duty cycle)
                new_half_period = new_period / 2.0

                # Format with 6 decimal places
                new_period_str = f"{new_period:.6f}"
                new_half_period_str = f"{new_half_period:.6f}"

                # Reconstruct the line based on pattern type
                if pattern_type == 1:
                    # Pattern 1 format
                    new_line = (
                        f"{match.group(1)}{clock_name}{match.group(3)}"
                        f"{new_period_str}{match.group(5)}{match.group(6)}"
                        f"{match.group(7)}{new_half_period_str}"
                        f"{match.group(9)}"
                    )
                else:
                    # Pattern 2 format
                    new_line = (
                        f"{match.group(1)}{clock_name}{match.group(3)}"
                        f"{new_period_str}{match.group(5)}{match.group(6)}"
                        f"{match.group(7)}{new_half_period_str}"
                        f"{match.group(9)}{match.group(10)}{match.group(11)}"
                    )

                # Preserve any trailing content (newline, comments)
                if match.end() < len(line):
                    new_line += line[match.end():]
                else:
                    new_line += '\n'

                modified_lines.append(new_line)
                modifications_count += 1

                unit = "ps" if is_ps else "ns"
                adj_val = adjustment_value_ps if is_ps else (
                    adjustment_value_ps / 1000.0
                )
                print(f"Modified {clock_name_stripped}: "
                      f"{period:.6f}{unit} -> {new_period:.6f}{unit} "
                      f"(adjustment: {adj_val:+.4f}{unit})")
            else:
                # Not a target clock, keep line unchanged
                modified_lines.append(line)
        else:
            # Not a create_clock line, keep unchanged
            modified_lines.append(line)

    # Write output file
    with open(output_file, 'w') as f:
        f.writelines(modified_lines)

    print(f"\nProcessed {input_file}")
    print(f"Modified {modifications_count} clock(s)")
    print(f"Output written to: {output_file}")


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]

    try:
        adjustment = int(sys.argv[2])
    except ValueError:
        print(f"Error: Adjustment parameter must be an integer")
        print(__doc__)
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    try:
        adjust_sdc_clock(input_file, adjustment)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
