#!/bin/bash

# Find and replace the massive inline script with module imports

INPUT_FILE="live_monitor_ui.html"
OUTPUT_FILE="live_monitor_ui_modular.html"

# Get line numbers
SCRIPT_START=$(grep -n "^    <script>$" "$INPUT_FILE" | head -1 | cut -d: -f1)
SCRIPT_END=$(grep -n "^    </script>$" "$INPUT_FILE" | grep -A1 "$SCRIPT_START" | tail -1 | cut -d: -f1)

echo "Script starts at line: $SCRIPT_START"
echo "Script ends at line: $SCRIPT_END"

# Create new file with everything before the inline script
head -n $((SCRIPT_START - 1)) "$INPUT_FILE" > "$OUTPUT_FILE"

# Add module imports
cat >> "$OUTPUT_FILE" << 'EOF'

    <!-- ========================================
         MODULAR JAVASCRIPT - REFACTORED
         All inline code extracted to modules
         ======================================== -->

    <!-- Load modules (ES6) -->
    <script type="module" src="js/main.js"></script>

    <!-- Keep existing modular files for compatibility -->
    <script src="monitor_metrics.js"></script>
    <script src="monitor_charts.js"></script>
    <script src="monitor_improvements.js"></script>

    <!-- Fallback for remaining inline functions (temporary) -->
    <script>
        // Global utility functions still used by inline HTML event handlers
        // TODO: Migrate these to event listeners in modules

        function closeShortcutsModal() {
            document.getElementById('shortcutsModal').style.display = 'none';
        }

        function openShortcutsModal() {
            document.getElementById('shortcutsModal').style.display = 'flex';
        }

        // These will be handled by the main module
        console.log('⚡ Modular Training Monitor Loading...');
    </script>

EOF

# Add everything after the inline script
tail -n +$((SCRIPT_END + 1)) "$INPUT_FILE" >> "$OUTPUT_FILE"

echo "✅ Created modular version: $OUTPUT_FILE"
