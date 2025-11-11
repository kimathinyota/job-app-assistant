// frontend/src/components/applications/IntelligentTextArea.jsx
import React, { useState, useEffect, useRef, useMemo } from 'react';

// --- CSS for the component ---
const STYLES = `
.intelligent-textarea-wrapper {
    position: relative;
    width: 100%;
}

.intelligent-editor {
    min-height: 80px;
    width: 100%;
    padding: 0.375rem 0.75rem;
    font-size: 0.9rem;
    font-weight: 400;
    line-height: 1.5;
    color: var(--bs-body-color);
    background-color: var(--bs-body-bg);
    border: 1px solid var(--bs-border-color);
    border-radius: 0.375rem;
    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    overflow-y: auto;
    white-space: pre-wrap;
}

.intelligent-editor:focus {
    color: var(--bs-body-color);
    background-color: var(--bs-body-bg);
    border-color: #86b7fe;
    outline: 0;
    box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25);
}

.intelligent-editor[data-placeholder="true"]:before {
    content: 'Click to add notes... Type @ to reference your CV...';
    color: var(--bs-gray-500);
    cursor: text;
}

.intelligent-editor a {
    color: var(--bs-primary);
    text-decoration: none;
    font-weight: 500;
    background-color: var(--bs-primary-bg-subtle);
    padding: 1px 4px;
    border-radius: 4px;
    cursor: pointer;
}

.suggestions-popup {
    position: absolute;
    z-index: 1050; /* Above modals */
    background: var(--bs-white);
    border: 1px solid var(--bs-border-color);
    border-radius: 0.375rem;
    box-shadow: var(--bs-box-shadow-lg);
    max-height: 250px;
    overflow-y: auto;
    width: 300px;
}

.suggestion-item {
    display: block;
    width: 100%;
    padding: 0.5rem 1rem;
    clear: both;
    font-weight: 400;
    color: var(--bs-body-color);
    text-align: inherit;
    text-decoration: none;
    white-space: nowrap;
    background-color: transparent;
    border: 0;
    cursor: pointer;
}

.suggestion-item:hover, .suggestion-item.active {
    background-color: var(--bs-primary-bg-subtle);
    color: var(--bs-primary-text-emphasis);
}

.suggestion-item small {
    display: block;
    color: var(--bs-text-muted);
}
`;

// Regex to find our custom references
const REFERENCE_REGEX = /\[(.*?)]<:(.*?)><(.*?)>/g;

const IntelligentTextArea = ({ initialValue, onSave, cv, onShowPreview }) => {
    const editorRef = useRef(null);
    const [rawText, setRawText] = useState(initialValue || '');
    const [isFocused, setIsFocused] = useState(false);
    
    // --- Suggestion State ---
    const [suggestions, setSuggestions] = useState({
        open: false,
        query: '',
        type: null,
        items: [],
        position: { top: 0, left: 0 },
        activeIndex: 0,
    });

    // --- Flatten CV data into a searchable structure ---
    const cvCategories = useMemo(() => {
        if (!cv) return [];
        return [
            { type: 'experiences', name: 'Experiences', items: cv.experiences || [] },
            { type: 'projects', name: 'Projects', items: cv.projects || [] },
            { type: 'skills', name: 'Skills', items: cv.skills || [] },
            { type: 'education', name: 'Education', items: cv.education || [] },
            { type: 'achievements', name: 'Achievements', items: cv.achievements || [] },
            { type: 'hobbies', name: 'Hobbies', items: cv.hobbies || [] },
        ].filter(cat => cat.items.length > 0);
    }, [cv]);

    // --- Convert raw text to display HTML ---
    const parseRawToHtml = (text) => {
        if (!text) return '';
        // Use a function for replacement to correctly handle special chars
        return text.replace(REFERENCE_REGEX, (match, type, id, name) => {
            const el = document.createElement('a');
            el.href = "#";
            el.dataset.type = type;
            el.dataset.id = id;
            el.setAttribute("contenteditable", "false");
            el.textContent = name;
            return el.outerHTML;
        });
    };

    // --- Convert display HTML back to raw text ---
    const parseHtmlToRaw = (element) => {
        let raw = '';
        element.childNodes.forEach(node => {
            if (node.nodeType === Node.TEXT_NODE) {
                raw += node.textContent;
            } else if (node.nodeType === Node.ELEMENT_NODE && node.tagName === 'A') {
                const type = node.dataset.type;
                const id = node.dataset.id;
                const name = node.textContent;
                raw += `[${type}]<:${id}><${name}>`;
            } else if (node.nodeType === Node.ELEMENT_NODE) {
                // Handle newlines from <div> or <br>
                if (node.tagName === 'BR') {
                    raw += '\n';
                } else if (node.tagName === 'DIV') {
                    if (raw.length > 0) raw += '\n'; // Add newline before div content
                    raw += parseHtmlToRaw(node);
                } else {
                    raw += parseHtmlToRaw(node); // Recurse for other nodes
                }
            }
        });
        return raw;
    };

    // --- Sync editor display when initialValue changes ---
    useEffect(() => {
        if (editorRef.current && !isFocused) {
            const newRawText = initialValue || '';
            setRawText(newRawText);
            editorRef.current.innerHTML = parseRawToHtml(newRawText);
        }
    }, [initialValue]);

    // --- Handle editor blur (save) ---
    const handleBlur = () => {
        setIsFocused(false);
        setSuggestions({ open: false, items: [], query: '', type: null, position: {}, activeIndex: 0 });
        const currentRawText = parseHtmlToRaw(editorRef.current);
        
        if (currentRawText !== initialValue) {
            onSave(currentRawText);
        }
        // Restore placeholder if empty
        if (editorRef.current && !currentRawText.trim()) {
            editorRef.current.dataset.placeholder = "true";
            editorRef.current.innerHTML = ""; // Clear any stray <br>
        }
    };

    // --- Handle editor focus ---
    const handleFocus = () => {
        setIsFocused(true);
        if (editorRef.current) {
            editorRef.current.dataset.placeholder = "false";
        }
    };

    // --- Get caret position (to place popup) ---
    const getCaretCoordinates = () => {
        const sel = window.getSelection();
        if (sel.rangeCount > 0) {
            const range = sel.getRangeAt(0).cloneRange();
            // Create a dummy span to get coords
            range.collapse(true);
            const span = document.createElement('span');
            range.insertNode(span);
            const rect = span.getBoundingClientRect();
            const { top, left } = rect;
            span.parentNode.removeChild(span); // Clean up
            
            // Fallback if rect is 0,0 (e.g., empty line)
            if (top === 0 && left === 0) {
                const editorRect = editorRef.current.getBoundingClientRect();
                return { top: editorRect.top + window.scrollY, left: editorRect.left + window.scrollX };
            }
            return { top: rect.bottom + window.scrollY, left: rect.left + window.scrollX };
        }
        return {};
    };

    // --- Handle click on links ---
    const handleClick = (e) => {
        if (e.target.tagName === 'A' && e.target.dataset.type) {
            e.preventDefault();
            const { type, id } = e.target.dataset;
            const category = cvCategories.find(c => c.type === type);
            if (!category) return;
            
            const item = category.items.find(i => i.id === id);
            if (item) {
                onShowPreview(item, type);
            }
        }
    };

    // --- Main input handler (the magic) ---
    const handleInput = () => {
        const selection = window.getSelection();
        const node = selection.focusNode;
        const offset = selection.focusOffset;
        
        if (!node || node.nodeType !== Node.TEXT_NODE) {
            setSuggestions(prev => ({ ...prev, open: false }));
            return;
        }

        const text = node.textContent.substring(0, offset);
        const atMatch = text.match(/@([\w\s]*)$/);

        if (atMatch) {
            const query = atMatch[1].toLowerCase();
            let items = [];
            let type = suggestions.type;

            if (!type) {
                // Stage 1: Filter categories
                items = cvCategories.filter(cat => cat.name.toLowerCase().includes(query));
            } else {
                // Stage 2: Filter items within a category
                const category = cvCategories.find(c => c.type === type);
                items = category.items.filter(item => 
                    (item.name || item.title || item.degree || item.company)
                        .toLowerCase().includes(query)
                );
            }
            
            setSuggestions({
                open: true,
                query: query,
                type: type,
                items: items,
                position: getCaretCoordinates(),
                activeIndex: 0,
            });
        } else {
            setSuggestions(prev => ({ ...prev, open: false, type: null })); // Close popup and reset type
        }
    };

    // --- Insert reference into editor ---
    const insertReference = (item, type) => {
        const selection = window.getSelection();
        if (!selection.rangeCount) return;
        const range = selection.getRangeAt(0);
        
        // 1. Find text node and delete the "@query"
        const node = selection.focusNode;
        const offset = selection.focusOffset;
        // Adjust query length based on whether we are in stage 1 or 2
        const queryLength = suggestions.type 
            ? suggestions.query.length + 1 // @ + query
            : suggestions.query.length + 1; // @ + category query (which is now part of the text)

        if (node.nodeType === Node.TEXT_NODE) {
            range.setStart(node, offset - queryLength);
            range.setEnd(node, offset);
        }
        range.deleteContents();

        // 2. Create the link element
        const link = document.createElement('a');
        const itemName = item.name || item.title || item.degree || item.company;
        link.href = "#";
        link.dataset.type = type;
        link.dataset.id = item.id;
        link.setAttribute("contenteditable", "false");
        link.textContent = itemName;

        // 3. Insert the link
        range.insertNode(link);

        // 4. Move cursor after the link
        range.setStartAfter(link);
        range.setEndAfter(link);
        selection.removeAllRanges();
        selection.addRange(range);

        // 5. Add a space after for easy typing
        const space = document.createTextNode(' ');
        range.insertNode(space);
        range.setStartAfter(space);
        range.setEndAfter(space);
        selection.removeAllRanges();
        selection.addRange(range);

        // 6. Close popup
        setSuggestions({ open: false, items: [], query: '', type: null, position: {}, activeIndex: 0 });
        editorRef.current.focus();
    };

    // --- Handle suggestion selection ---
    const selectSuggestion = (item) => {
        if (!suggestions.type) {
            // Stage 1: Category selected, move to Stage 2
            const category = cvCategories.find(c => c.type === item.type);
            setSuggestions(prev => ({
                ...prev,
                type: item.type,
                query: '', // Reset query
                items: category.items,
                activeIndex: 0,
            }));
            
            // --- Replace "@Category" with "@" to start searching
            const selection = window.getSelection();
            const range = selection.getRangeAt(0);
            const node = selection.focusNode;
            const offset = selection.focusOffset;
            const queryLength = suggestions.query.length;
            
            range.setStart(node, offset - queryLength);
            range.setEnd(node, offset);
            range.deleteContents();
            
        } else {
            // Stage 2: Item selected, insert it
            insertReference(item, suggestions.type);
        }
    };

    // --- Keyboard navigation for suggestions ---
    const handleKeyDown = (e) => {
        if (!suggestions.open) return;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setSuggestions(prev => ({
                ...prev,
                activeIndex: (prev.activeIndex + 1) % prev.items.length,
            }));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setSuggestions(prev => ({
                ...prev,
                activeIndex: (prev.activeIndex - 1 + prev.items.length) % prev.items.length,
            }));
        } else if (e.key === 'Enter' || e.key === 'Tab') {
            e.preventDefault();
            if (suggestions.items.length > 0) {
                selectSuggestion(suggestions.items[suggestions.activeIndex]);
            }
        } else if (e.key === 'Backspace' && suggestions.query === '' && suggestions.type) {
             e.preventDefault();
             // Go back to category selection
             setSuggestions(prev => ({
                ...prev,
                type: null,
                query: '',
                items: cvCategories,
                activeIndex: 0,
             }));
        } else if (e.key === 'Escape') {
            e.preventDefault();
            setSuggestions(prev => ({ ...prev, open: false, type: null }));
        }
    };

    const getItemName = (item, type) => {
        switch(type) {
            case 'experiences': return item.title;
            case 'projects': return item.name;
            case 'skills': return item.name;
            case 'education': return item.degree;
            case 'achievements': return item.name;
            case 'hobbies': return item.name;
            default: return 'Unknown';
        }
    }
    
    const getItemSubtitle = (item, type) => {
        switch(type) {
            case 'experiences': return item.company;
            case 'education': return item.institution;
            default: return null;
        }
    }

    return (
        <div className="intelligent-textarea-wrapper">
            <style>{STYLES}</style>
            <div
                ref={editorRef}
                className="intelligent-editor"
                contentEditable="true"
                onFocus={handleFocus}
                onBlur={handleBlur}
                onInput={handleInput}
                onKeyDown={handleKeyDown}
                onClick={handleClick}
                dangerouslySetInnerHTML={{ __html: parseRawToHtml(rawText) }}
                data-placeholder={!rawText && !isFocused}
            />
            {suggestions.open && (
                <div
                    className="suggestions-popup"
                    style={{
                        top: suggestions.position.top + 5,
                        left: suggestions.position.left,
                    }}
                >
                    {suggestions.items.length === 0 && (
                        <span className="suggestion-item text-muted">No results found</span>
                    )}
                    {suggestions.items.map((item, index) => {
                        const name = suggestions.type ? getItemName(item, suggestions.type) : item.name;
                        const subtitle = suggestions.type ? getItemSubtitle(item, suggestions.type) : null;
                        
                        return (
                            <button
                                key={item.id || item.type}
                                type="button"
                                className={`suggestion-item ${index === suggestions.activeIndex ? 'active' : ''}`}
                                onClick={() => selectSuggestion(item)}
                            >
                                {name}
                                {subtitle && <small>{subtitle}</small>}
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

export default IntelligentTextArea;