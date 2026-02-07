/**
 * Generates a display name for a CV based on available fields.
 * Logic: "Title First Last" (e.g. "Senior Dev John Doe")
 * Fallback: "InternalName (Title)" or just "InternalName"
 */
export const getCVDisplayName = (cv) => {
    if (!cv) return "Untitled CV";
    
    const { name, first_name, last_name, title } = cv;
    
    // Check if we have actual name parts
    const hasRealName = (first_name && first_name.trim()) || (last_name && last_name.trim());
    
    if (hasRealName) {
        // Strategy A: Build "Title First Last"
        // Filter out null/empty values to avoid extra spaces
        return [title, first_name, last_name]
            .filter(part => part && part.trim())
            .join(' ');
    } else {
        // Strategy B: Use Internal Name with Title fallback
        // e.g. "My Master CV (Senior Dev)"
        return (title && title.trim()) ? `${name} (${title})` : name;
    }
};


/**
 * NEW: Formats a date string into "Month Year" (e.g. "Jan 2023")
 * Handles YYYY-MM-DD, YYYY-MM, or standard JS date strings.
 */
export const formatMonthYear = (dateStr) => {
    if (!dateStr) return null;
    const date = new Date(dateStr);
    if (isNaN(date.getTime())) return dateStr; // Return original if parsing fails (fallback)
    
    return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
};

/**
 * NEW: Formats a date range with intelligent logic.
 * - Standard: "Jan 2020 – Feb 2021"
 * - Present: "Jan 2020 – Present"
 * - Same Month/Year: 
 * - If days differ: "Jan 12 – 15, 2023"
 * - If no days: "Jan 2023"
 */
export const formatDateRange = (startStr, endStr) => {
    if (!startStr) return "";

    const start = new Date(startStr);
    // If endStr is null/empty, assume "Present"
    const isPresent = !endStr;
    const end = isPresent ? new Date() : new Date(endStr);

    // If parsing fails, fallback to simple string concatenation
    if (isNaN(start.getTime()) || (!isPresent && isNaN(end.getTime()))) {
        return `${startStr} – ${endStr || 'Present'}`;
    }

    // Check if same month and year
    const sameMonth = start.getMonth() === end.getMonth();
    const sameYear = start.getFullYear() === end.getFullYear();

    if (!isPresent && sameMonth && sameYear) {
        // They are in the same month. Do we have day precision?
        // A simple heuristic: if the input string string length > 7 (YYYY-MM is 7 chars), we likely have days.
        const hasDays = startStr.length > 7 && endStr.length > 7;
        
        if (hasDays) {
            // "Jan 12 – 15, 2020"
            const startDay = start.getDate();
            const endDay = end.getDate();
            const monthYear = start.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
            return `${start.toLocaleDateString('en-US', { month: 'short' })} ${startDay} – ${endDay}, ${start.getFullYear()}`;
        } else {
            // Just "Jan 2020"
            return formatMonthYear(startStr);
        }
    }

    // Standard Range
    const startDisplay = formatMonthYear(startStr);
    const endDisplay = isPresent ? "Present" : formatMonthYear(endStr);
    return `${startDisplay} – ${endDisplay}`;
};