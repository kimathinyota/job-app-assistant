// frontend/src/components/applications/IntelligentEditorBase.jsx
// This is the new base component that powers both editors.
import React, { useState, useEffect, useRef, useMemo, forwardRef } from 'react';

// Regex to find our custom references
const REFERENCE_REGEX = /\[(.*?)]<:(.*?)><(.*?)>/g;

// Helper to get a simple name for a CV item
const getItemName = (item, type) => {
    switch(type) {
        case 'experiences': return item.title;
        case 'projects': return item.title || item.name;
        case 'skills': return item.name;
        case 'education': return item.degree;
        case 'achievements': return item.text;
        case 'hobbies': return item.name;
        default: return 'Unknown';
    }
};

const getItemSubtitle = (item, type) => {
    switch(type) {
        case 'experiences': return item.company;
        case 'education': return item.institution;
        default: return null;
    }
};

const IntelligentEditorBase = ({
    initialValue,
    onSave,
    fullCV,
    jobFeatures = [],
    onShowPreview,
    // --- Configuration Props ---
    enableAtLinking = true,
    enableSlashCommands = false,
    enableStrategyRail = false,
    focusedIdea = null,
    placeholder = "Start writing...",
    minHeight = "100px"
}) => {
    const editorRef = useRef(null);
    const [isFocused, setIsFocused] = useState(false);
    const suggestionQueryRef = useRef(null); // Stores { node, offset }
    
    const [suggestions, setSuggestions] = useState({
        open: false,
        type: null, // '@cv', '@evidence', '/command'
        category: null, // e.g., 'experiences'
        items: [],
        position: { top: 0, left: 0 },
        activeIndex: 0,
    });

    // --- Data Memos ---
    const cvCategories = useMemo(() => {
        if (!fullCV) return [];
        return [
            { type: 'experiences', name: 'Experiences', items: fullCV.experiences || [] },
            { type: 'projects', name: 'Projects', items: fullCV.projects || [] },
            { type: 'skills', name: 'Skills', items: fullCV.skills || [] },
            { type: 'education', name: 'Education', items: fullCV.education || [] },
            { type: 'achievements', name: 'Achievements', items: fullCV.achievements || [] },
            { type: 'hobbies', name: 'Hobbies', items: fullCV.hobbies || [] },
        ];
    }, [fullCV]);

    const jobFeatureItems = useMemo(() => {
        return jobFeatures.map(f => ({
            id: f.id,
            name: f.description,
            type: f.type,
        }));
    }, [jobFeatures]);
    
    // --- Parsing ---
    const parseRawToHtml = (text) => {
        if (!text) return '';
        return text.replace(REFERENCE_REGEX, (match, type, id, name) => {
            return `<a href="#" data-type="${type}" data-id="${id}" contenteditable="false">${name}</a>`;
        });
    };

    const parseHtmlToRaw = (element) => {
        let raw = '';
        if (!element) return '';
        element.childNodes.forEach(node => {
            if (node.nodeType === Node.TEXT_NODE) {
                raw += node.textContent;
            } else if (node.nodeType === Node.ELEMENT_NODE && node.tagName === 'A') {
                raw += `[${node.dataset.type}]<:${node.dataset.id}><${node.textContent}>`;
            } else if (node.nodeBody === Node.ELEMENT_NODE) {
                if (node.tagName === 'BR') raw += '\n';
                else if (node.tagName === 'DIV') {
                    if (raw.length > 0 && !raw.endsWith('\n')) raw += '\n';
                    raw += parseHtmlToRaw(node);
                } else {
                    raw += parseHtmlToRaw(node);
                }
            }
        });
        return raw;
    };

    // --- Effects ---
    useEffect(() => {
        if (editorRef.current && !isFocused) {
            const newRawText = initialValue || '';
            editorRef.current.innerHTML = parseRawToHtml(newRawText);
            editorRef.current.dataset.placeholder = !newRawText;
        }
    }, [initialValue, isFocused]);

    // --- Handlers ---
    const handleFocus = () => {
        setIsFocused(true);
        if (editorRef.current) editorRef.current.dataset.placeholder = "false";
    };

    const handleBlur = () => {
        setIsFocused(false);
        setTimeout(() => {
             if (!isFocused) setSuggestions(prev => ({ ...prev, open: false }));
        }, 150);
        
        const currentRawText = parseHtmlToRaw(editorRef.current);
        if (currentRawText !== initialValue) {
            onSave(currentRawText);
        }
        
        if (editorRef.current) {
            editorRef.current.dataset.placeholder = !currentRawText.trim();
        }
    };

    const getCaretCoordinates = () => {
        const sel = window.getSelection();
        if (sel.rangeCount === 0) return { top: 0, left: 0 };
        const range = sel.getRangeAt(0).cloneRange();
        range.collapse(true);
        const span = document.createElement('span');
        range.insertNode(span);
        const rect = span.getBoundingClientRect();
        const editorRect = editorRef.current.getBoundingClientRect();
        span.parentNode.removeChild(span);
        return { 
            top: rect.bottom - editorRect.top, 
            left: rect.left - editorRect.left 
        };
    };

    const handleClick = (e) => {
        if (e.target.tagName === 'A' && e.target.dataset.type && onShowPreview) {
            e.preventDefault();
            const { type, id } = e.target.dataset;
            const category = cvCategories.find(c => c.type === type);
            if (!category) return;
            const item = category.items.find(i => i.id === id);
            if (item) {
                onShowPreview({ item, type });
            }
        }
    };

    const handleInput = () => {
        const selection = window.getSelection();
        if (!selection.rangeCount) return;
        const range = selection.getRangeAt(0);
        if (!range.collapsed) return;
        
        const node = range.startContainer;
        const offset = range.startOffset;

        if (editorRef.current) {
            editorRef.current.dataset.placeholder = !editorRef.current.textContent.trim();
        }
        
        if (!node || node.nodeType !== Node.TEXT_NODE) {
            setSuggestions(prev => ({ ...prev, open: false }));
            return;
        }

        const text = node.textContent.substring(0, offset);
        
        // --- @ CV Linking Logic ---
        const atMatch = enableAtLinking && text.match(/@([\w\s]*)$/);
        if (atMatch) {
            const query = atMatch[1].toLowerCase();
            let items = [];
            
            // Strategy Rail logic
            if (enableStrategyRail && focusedIdea && focusedIdea.mapping_pair_ids.length > 0) {
                // Pre-filtered evidence mode
                // We need the full pair objects, not just IDs. This needs a lookup.
                // This part is complex and needs `pairMap` passed down.
                // For now, we'll just show all CV items.
                items = cvCategories.filter(c => c.items.length > 0 && c.name.toLowerCase().includes(query));
            } else {
                // Standard mode
                items = cvCategories.filter(c => c.items.length > 0 && c.name.toLowerCase().includes(query));
            }

            suggestionQueryRef.current = { node, offset: offset - query.length - 1 };
            setSuggestions({
                open: true, type: '@cv', category: null,
                items: items, position: getCaretCoordinates(), activeIndex: 0
            });
            return;
        }

        // --- / Command Logic ---
        const slashMatch = enableSlashCommands && text.match(/\/([\w\s]*)$/);
        if (slashMatch) {
            const query = slashMatch[1].toLowerCase();
            const items = jobFeatureItems.filter(f => f.name.toLowerCase().includes(query));
            
            suggestionQueryRef.current = { node, offset: offset - query.length - 1 };
            setSuggestions({
                open: true, type: '/command', category: null,
                items: items, position: getCaretCoordinates(), activeIndex: 0
            });
            return;
        }
        
        // --- Sub-query (e.g., @Experiences ...) ---
        if (suggestions.open && suggestions.type === '@cv' && suggestions.category) {
            const queryStartOffset = suggestionQueryRef.current?.offset + 1; // +1 for the @
            const query = text.substring(queryStartOffset).toLowerCase();
            const category = cvCategories.find(c => c.type === suggestions.category);
            if (!category) {
                setSuggestions(prev => ({ ...prev, open: false })); return;
            }
            const items = category.items.filter(item => 
                getItemName(item, category.type).toLowerCase().includes(query)
            );
            setSuggestions(prev => ({ ...prev, items, position: getCaretCoordinates(), activeIndex: 0 }));
            return;
        }

        setSuggestions(prev => ({ ...prev, open: false }));
    };

    const insertReference = (item, type) => {
        const selection = window.getSelection();
        if (!selection.rangeCount || !selection.focusNode || !suggestionQueryRef.current) return;
        
        const range = selection.getRangeAt(0);
        const { node, offset } = suggestionQueryRef.current;
        
        // Delete the trigger text (e.g., "@React" or "/Requirement")
        range.setStart(node, offset); 
        range.setEnd(selection.focusNode, selection.focusOffset);
        range.deleteContents();

        const link = document.createElement('a');
        link.href = "#";
        link.dataset.type = type;
        link.dataset.id = item.id;
        link.setAttribute("contenteditable", "false");
        link.textContent = getItemName(item, type);

        range.insertNode(link);
        range.setStartAfter(link);
        range.setEndAfter(link);
        selection.removeAllRanges();
        selection.addRange(range);

        const space = document.createTextNode(' ');
        range.insertNode(space);
        range.setStartAfter(space);
        selection.removeAllRanges();
        selection.addRange(range);

        setSuggestions({ open: false, items: [], type: null, position: {}, activeIndex: 0 });
        editorRef.current.focus();
    };

    const selectSuggestion = (item) => {
        const { type } = suggestions;
        
        if (type === '@cv' && !suggestions.category) {
            // Stage 1: Category selected
            const category = cvCategories.find(c => c.type === item.type);
            const selection = window.getSelection();
            if (!selection.rangeCount || !selection.focusNode) return;
            const range = selection.getRangeAt(0);
            
            // Delete the category query text
            const { node, offset } = suggestionQueryRef.current;
            range.setStart(node, offset + 1); // After the @
            range.setEnd(selection.focusNode, selection.focusOffset);
            range.deleteContents();
            
            setSuggestions(prev => ({
                ...prev,
                category: item.type,
                items: category.items,
                activeIndex: 0,
                position: getCaretCoordinates(),
            }));
        } else if (type === '@cv' && suggestions.category) {
            // Stage 2: Item selected
            insertReference(item, suggestions.category);
        } else if (type === '/command') {
            // /command is just a shortcut for @linking
            insertReference(item, item.type); // type is 'requirement', 'value' etc.
        }
    };

    const handleKeyDown = (e) => {
        if (!suggestions.open || suggestions.items.length === 0) return;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setSuggestions(prev => ({ ...prev, activeIndex: (prev.activeIndex + 1) % prev.items.length }));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setSuggestions(prev => ({ ...prev, activeIndex: (prev.activeIndex - 1 + prev.items.length) % prev.items.length }));
        } else if (e.key === 'Enter' || e.key === 'Tab') {
            e.preventDefault();
            selectSuggestion(suggestions.items[suggestions.activeIndex]);
        } else if (e.key === 'Backspace' && suggestions.type === '@cv' && suggestions.category) {
             // This logic needs to be smarter to check *what* was deleted
             // For now, we'll just close
             setSuggestions(prev => ({ ...prev, open: false }));
        } else if (e.key === 'Escape') {
            e.preventDefault();
            setSuggestions(prev => ({ ...prev, open: false }));
        }
    };

    return (
        <div className="intelligent-editor-wrapper position-relative">
            <style>{`
                .intelligent-editor {
                    min-height: ${minHeight};
                    width: 100%;
                    padding: 0.5rem 0.75rem;
                    font-size: 0.9rem;
                    line-height: 1.6;
                    color: #212529;
                    background-color: #fff;
                    border: 1px solid #dee2e6;
                    border-radius: 0.375rem;
                    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
                    overflow-y: auto;
                    white-space: pre-wrap;
                }
                .intelligent-editor:focus {
                    color: #212529;
                    background-color: #fff;
                    border-color: #86b7fe;
                    outline: 0;
                    box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25);
                }
                .intelligent-editor[data-placeholder="true"]:before {
                    content: '${placeholder}';
                    color: #6c757d;
                    cursor: text;
                    pointer-events: none;
                }
                .intelligent-editor a {
                    color: #0d6efd;
                    text-decoration: none;
                    font-weight: 500;
                    background-color: rgba(13, 110, 253, 0.1);
                    padding: 1px 4px;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .suggestions-popup {
                    position: absolute;
                    z-index: 1050;
                    background: #fff;
                    border: 1px solid #dee2e6;
                    border-radius: 0.375rem;
                    box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
                    max-height: 250px;
                    overflow-y: auto;
                    width: 300px;
                }
                .suggestion-item {
                    display: block;
                    width: 100%;
                    padding: 0.5rem 1rem;
                    font-weight: 400;
                    color: #212529;
                    text-align: inherit;
                    text-decoration: none;
                    white-space: nowrap;
                    background-color: transparent;
                    border: 0;
                    cursor: pointer;
                }
                .suggestion-item:hover, .suggestion-item.active {
                    background-color: #f8f9fa;
                }
                .suggestion-item small {
                    display: block;
                    color: #6c757d;
                }
            `}</style>
            
            <div
                ref={editorRef}
                className="intelligent-editor"
                contentEditable="true"
                onFocus={handleFocus}
                onBlur={handleBlur}
                onInput={handleInput}
                onKeyDown={handleKeyDown}
                onClick={handleClick}
                data-placeholder={!initialValue && !isFocused}
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
                        const name = (suggestions.type === '/command') 
                            ? item.name 
                            : getItemName(item, suggestions.category || item.type);
                        const subtitle = (suggestions.type === '/command')
                            ? item.type
                            : getItemSubtitle(item, suggestions.category || item.type);
                        
                        return (
                            <button
                                key={item.id || item.type}
                                type="button"
                                className={`suggestion-item ${index === suggestions.activeIndex ? 'active' : ''}`}
                                onMouseDown={(e) => { // Use onMouseDown to prevent blur
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

export default IntelligentEditorBase;