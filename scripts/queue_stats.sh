#!/bin/bash
#
# Queue Statistics - View data in all queues
#

cd /path/to/training/queue

echo "================================================================================"
echo "📊 TRAINING DATA QUEUE STATISTICS"
echo "================================================================================"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

TOTAL_FILES=0
TOTAL_EXAMPLES=0

for dir in */; do
    if [ -d "$dir" ]; then
        FILES=$(find "$dir" -name "*.jsonl" -type f 2>/dev/null | wc -l)
        EXAMPLES=$(cat "$dir"*.jsonl 2>/dev/null | wc -l)
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        
        TOTAL_FILES=$((TOTAL_FILES + FILES))
        TOTAL_EXAMPLES=$((TOTAL_EXAMPLES + EXAMPLES))
        
        printf "%-25s: %4d files, %8d examples, %8s\n" "${dir%/}" $FILES $EXAMPLES $SIZE
    fi
done

echo "--------------------------------------------------------------------------------"
printf "%-25s: %4d files, %8d examples\n" "TOTAL" $TOTAL_FILES $TOTAL_EXAMPLES
echo "================================================================================"
echo ""

# Show recent activity
echo "📁 Recent Files (last 5):"
find . -name "*.jsonl" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -5 | while read timestamp file; do
    date_str=$(date -d @${timestamp%.*} '+%Y-%m-%d %H:%M:%S')
    size=$(stat -f "%z" "$file" 2>/dev/null || stat -c "%s" "$file" 2>/dev/null)
    lines=$(wc -l < "$file" 2>/dev/null)
    printf "  %s - %s (%s, %d examples)\n" "$date_str" "$file" "$(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo ${size}B)" "$lines"
done

echo ""
echo "================================================================================"
