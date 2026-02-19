/**
 * Generates a display name for a CV based on available fields.
 */
export const getCVDisplayName = (cv) => {
    if (!cv) return "Untitled CV";
    const { name, first_name, last_name, title } = cv;
    const hasRealName = (first_name && first_name.trim()) || (last_name && last_name.trim());
    
    if (hasRealName) {
        return [title, first_name, last_name].filter(part => part && part.trim()).join(' ');
    } else {
        return (title && title.trim()) ? `${name} (${title})` : name;
    }
};

/**
 * Formats a date string into "Month Year" (e.g. "Jan 2023")
 */
export const formatMonthYear = (dateStr) => {
    if (!dateStr) return null;
    const date = new Date(dateStr);
    if (isNaN(date.getTime())) return dateStr; 
    return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
};

/**
 * Formats a date range with intelligent logic.
 */
export const formatDateRange = (startStr, endStr) => {
    // If both are missing, return empty
    if (!startStr && !endStr) return "";

    // If only the end date is provided, return just the end date
    if (!startStr && endStr) {
        const end = new Date(endStr);
        return isNaN(end.getTime()) ? endStr : formatMonthYear(endStr);
    }

    const start = new Date(startStr);
    const isPresent = !endStr;
    const end = isPresent ? new Date() : new Date(endStr);

    if (isNaN(start.getTime()) || (!isPresent && isNaN(end.getTime()))) {
        return `${startStr} - ${endStr || 'Present'}`;
    }

    const sameMonth = start.getMonth() === end.getMonth();
    const sameYear = start.getFullYear() === end.getFullYear();

    if (!isPresent && sameMonth && sameYear) {
        const hasDays = startStr.length > 7 && endStr.length > 7;
        if (hasDays) {
            const startDay = start.getDate();
            const endDay = end.getDate();
            return `${start.toLocaleDateString('en-US', { month: 'short' })} ${startDay} - ${endDay}, ${start.getFullYear()}`;
        }
        return formatMonthYear(startStr);
    }

    const startDisplay = formatMonthYear(startStr);
    const endDisplay = isPresent ? "Present" : formatMonthYear(endStr);
    return `${startDisplay} - ${endDisplay}`;
};
/**
 * NEW: Returns a human-readable relative time string.
 * e.g. "2 minutes ago", "1 hour ago"
 */
export const timeAgo = (dateInput) => {
    if (!dateInput) return '';
    
    const date = new Date(dateInput);
    // Handle UTC conversion if needed, though usually browser handles ISO strings correctly
    const now = new Date();
    const seconds = Math.floor((now - date) / 1000);

    // Future check (in case client/server clocks drift slightly)
    if (seconds < 0) return "Just now";

    let interval = seconds / 31536000;
    if (interval > 1) return Math.floor(interval) + " years ago";
    
    interval = seconds / 2592000;
    if (interval > 1) return Math.floor(interval) + " months ago";
    
    interval = seconds / 86400;
    if (interval > 1) return Math.floor(interval) + " days ago";
    
    interval = seconds / 3600;
    if (interval > 1) return Math.floor(interval) + " hours ago";
    
    interval = seconds / 60;
    if (interval > 1) return Math.floor(interval) + " minutes ago";
    
    return "Just now";
};