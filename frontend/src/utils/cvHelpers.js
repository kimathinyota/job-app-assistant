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