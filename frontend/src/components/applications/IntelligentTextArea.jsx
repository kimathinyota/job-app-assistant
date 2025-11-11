// frontend/src/components/applications/IntelligentTextArea.jsx
import React, { useState, useEffect, useRef, useMemo } from 'react';

// --- CSS for the component ---
const STYLES = `
.intelligent-textarea-wrapper {
    position: relative;
    width: 100%;
}

/* --- NEW: Style for the plus button --- */
.intelligent-textarea-plus-btn {
    position: absolute;
    top: 0px;
    right: 0px;
    z-index: 5;
    border: none;
    background: var(--bs-light);
    color: var(--bs-primary);
    border-radius: 0 0.375rem 0 0.375rem;
    font-size: 1.1rem;
    width: 30px;
    height: 30px;
    line-height: 1;
    opacity: 0.6;
    transition: opacity 0.15s ease-in-out;
}
.intelligent-textarea-plus-btn:hover {
    opacity: 1;
}
/* --- END NEW --- */


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
    /* --- NEW: Add padding to top-right to avoid button --- */
    padding-right: 35px;
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
    pointer-events: none;
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
    z-index: 1050;
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
    // This state is ONLY for setting the initial value and placeholder
    const [rawText, setRawText] = useState(initialValue || '');
    const [isFocused, setIsFocused] = useState(false);
    
    // This ref tracks the *start* of the @ query
    const suggestionQueryRef = useRef(null);
    
    // --- Suggestion State ---
    const [suggestions, setSuggestions] = useState({
        open: false,
        query: '',
        type: null, // 'category' or 'item'
        categoryType: null, // 'experiences', 'skills', etc.
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
                if (node.tagName === 'BR') {
                    raw += '\n';
                } else if (node.tagName === 'DIV') {
                    if (raw.length > 0 && !raw.endsWith('\n')) raw += '\n';
                    raw += parseHtmlToRaw(node);
                } else {
                    raw += parseHtmlToRaw(node);
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
            editorRef.current.dataset.placeholder = !newRawText;
        }
    }, [initialValue]);

    // --- Handle editor blur (save) ---
    const handleBlur = () => {
        setIsFocused(false);
        // Delay closing suggestions so a click on a suggestion can register
        setTimeout(() => {
             if (!isFocused) {
                setSuggestions(prev => ({ ...prev, open: false, type: null }));
             }
        }, 150);
        
        const currentRawText = parseHtmlToRaw(editorRef.current);
        setRawText(currentRawText); // Sync state

        if (currentRawText !== initialValue) {
            onSave(currentRawText);
        }
        
        if (editorRef.current) {
            editorRef.current.dataset.placeholder = !currentRawText.trim();
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
            range.collapse(true);
            const span = document.createElement('span');
            range.insertNode(span);
            const rect = span.getBoundingClientRect();
            const { top, left } = rect;
            span.parentNode.removeChild(span);
            
            const editorRect = editorRef.current.getBoundingClientRect();
            
            if (top === 0 && left === 0) {
                return { top: editorRect.top + window.scrollY, left: editorRect.left + window.scrollX };
            }
            // Position relative to the wrapper
            return { top: rect.bottom - editorRect.top, left: rect.left - editorRect.left };
        }
        return { top: 0, left: 0 };
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

    // --- Main input handler (THE FIX) ---
    const handleInput = () => {
        // We do NOT call setRawText here.
        const selection = window.getSelection();
        if (!selection.rangeCount) return;
        
        const range = selection.getRangeAt(0);
        if (!range.collapsed) return;
        
        const node = range.startContainer;
        const offset = range.startOffset;

        // Update placeholder state
        if (editorRef.current) {
            const hasText = editorRef.current.textContent.trim().length > 0;
            editorRef.current.dataset.placeholder = !hasText;
        }
        
        if (!node || node.nodeType !== Node.TEXT_NODE) {
            setSuggestions(prev => ({ ...prev, open: false }));
            return;
        }

        const text = node.textContent.substring(0, offset);
        
        // Check if we are *inside* a query
        if (suggestions.type) {
            const queryStartOffset = suggestionQueryRef.current?.offset || 0;
            const queryText = text.substring(queryStartOffset);
            
            const category = cvCategories.find(c => c.type === suggestions.categoryType);
            const items = category.items.filter(item => 
                (item.name || item.title || item.degree || item.company)
                    .toLowerCase().includes(queryText.toLowerCase())
            );
            
            setSuggestions(prev => ({
                ...prev,
                open: true,
                query: queryText,
                items: items,
                position: getCaretCoordinates(),
                activeIndex: 0,
            }));

        } else {
            // Check if we are *starting* a new query
            const atMatch = text.match(/@([\w\s]*)$/);
            if (atMatch) {
                const query = atMatch[1].toLowerCase();
                const items = cvCategories.filter(cat => cat.name.toLowerCase().includes(query));
                
                // Store the start of the query
                suggestionQueryRef.current = {
                    node: node,
                    offset: offset - query.length - 1 // -1 for '@'
                };
                
                setSuggestions({
                    open: true,
                    query: query,
                    type: 'category', // Stage 1: searching categories
                    categoryType: null,
                    items: items,
                    position: getCaretCoordinates(),
                    activeIndex: 0,
                });
            } else {
                setSuggestions(prev => ({ ...prev, open: false, type: null }));
            }
        }
    };

    // --- Insert reference ---
    const insertReference = (item, type) => {
        const selection = window.getSelection();
        if (!selection.rangeCount) return;
        
        const range = selection.getRangeAt(0);
        const { node, offset } = suggestionQueryRef.current;
        
        // Delete the entire query text ("@category query" or "@item query")
        range.setStart(node, offset);
        range.setEnd(selection.focusNode, selection.focusOffset);
        range.deleteContents();

        const link = document.createElement('a');
        const itemName = item.name || item.title || item.degree || item.company;
        link.href = "#";
        link.dataset.type = type;
        link.dataset.id = item.id;
        link.setAttribute("contenteditable", "false");
        link.textContent = itemName;

        range.insertNode(link);
        range.setStartAfter(link);
        range.setEndAfter(link);
        selection.removeAllRanges();
        selection.addRange(range);

        const space = document.createTextNode(' ');
        range.insertNode(space);
        range.setStartAfter(space);
        range.setEndAfter(space);
        selection.removeAllRanges();
        selection.addRange(range);

        setSuggestions({ open: false, items: [], query: '', type: null, position: {}, activeIndex: 0 });
        editorRef.current.focus();
        handleInput(); // Trigger placeholder update
    };

    // --- Handle suggestion selection ---
    const selectSuggestion = (item) => {
        const selection = window.getSelection();
        if (!selection.rangeCount) return;
        const range = selection.getRangeAt(0);

        if (suggestions.type === 'category') {
            // Stage 1: Category selected
            const category = cvCategories.find(c => c.type === item.type);
            
            // --- Replace "@Category" with "@" ---
            const { node, offset } = suggestionQueryRef.current;
            range.setStart(node, offset + 1); // After the "@"
            range.setEnd(selection.focusNode, selection.focusOffset);
            range.deleteContents();
            
            // Update suggestion state for Stage 2
            setSuggestions(prev => ({
                ...prev,
                type: 'item',
                categoryType: item.type,
                query: '',
                items: category.items,
                activeIndex: 0,
            }));
            
        } else {
            // Stage 2: Item selected
            insertReference(item, suggestions.categoryType);
        }
    };

    // --- Keyboard navigation ---
    const handleKeyDown = (e) => {
        if (!suggestions.open) return;
        if (suggestions.items.length === 0) return;

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
            selectSuggestion(suggestions.items[suggestions.activeIndex]);
        } else if (e.key === 'Backspace' && suggestions.query === '' && suggestions.type === 'item') {
             e.preventDefault();
             // Go back to category selection
             setSuggestions(prev => ({
                ...prev,
                type: 'category',
                categoryType: null,
                query: '',
                items: cvCategories,
                activeIndex: 0,
             }));
             suggestionQueryRef.current.offset = suggestionQueryRef.current.offset + 1; // Adjust ref
        } else if (e.key === 'Escape') {
            e.preventDefault();
            setSuggestions(prev => ({ ...prev, open: false, type: null }));
        }
    };
    
    // --- NEW: Handler for the plus button ---
    const handlePlusClick = () => {
        editorRef.current.focus();
        // Manually open the suggestions to Stage 1
        setSuggestions({
            open: true,
            query: '',
            type: 'category',
            categoryType: null,
            items: cvCategories,
            position: { top: 30, right: 5 }, // Position relative to wrapper
            activeIndex: 0,
        });
        // Set a dummy query ref
        suggestionQueryRef.current = { node: null, offset: -1 };
    };


    // --- Helper Functions (Unchanged) ---
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
            
            {/* --- NEW: Plus Button --- */}
            <button
                type="button"
                className="intelligent-textarea-plus-btn"
                title="Reference your CV"
                onMouseDown={(e) => {
                    // Use onMouseDown to prevent the editor from blurring
                    e.preventDefault(); 
                    handlePlusClick();
                }}
            >
                <i className="bi bi-plus-lg"></i>
            </button>
            
            <div
                ref={editorRef}
                className="intelligent-editor"
                contentEditable="true"
                onFocus={handleFocus}
                onBlur={handleBlur}
                onInput={handleInput}
                onKeyDown={handleKeyDown}
                onClick={handleClick}
                data-placeholder={!rawText && !isFocused}
                // We only set innerHTML on initial load via useEffect
            />
            {suggestions.open && (
                <div
                    className="suggestions-popup"
                    style={{
                        top: suggestions.position.top + 5,
                        left: suggestions.position.left,
                        right: suggestions.position.right ? suggestions.position.right + 5 : 'auto'
                    }}
                >
                    {suggestions.items.length === 0 && (
                        <span className="suggestion-item text-muted">No results found</span>
                    )}
                    {suggestions.items.map((item, index) => {
                        const name = suggestions.type === 'item' ? getItemName(item, suggestions.categoryType) : item.name;
                        const subtitle = suggestions.type === 'item' ? getItemSubtitle(item, suggestions.categoryType) : null;
                        
                        return (
                            <button
                                key={item.id || item.type}
                                type="button"
                                className={`suggestion-item ${index === suggestions.activeIndex ? 'active' : ''}`}
                                // Use onMouseDown to select *before* blur fires
                                onMouseDown={(e) => {
                                    e.preventDefault();
                                    selectSuggestion(item);
                                }}
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