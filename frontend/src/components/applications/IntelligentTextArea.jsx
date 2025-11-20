// frontend/src/components/applications/IntelligentTextArea.jsx
import React, { useEffect, useState, useMemo, forwardRef, useImperativeHandle } from 'react';
import { useEditor, EditorContent, ReactRenderer, mergeAttributes, NodeViewWrapper, NodeViewContent, ReactNodeViewRenderer } from '@tiptap/react';
import { Node, Mark, InputRule } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';
import Mention from '@tiptap/extension-mention';
import Link from '@tiptap/extension-link';
import {
    Ghost, Eye, EyeOff, Trash2, Heading1,
    Briefcase, GraduationCap, Cpu, Heart, Trophy, FolderGit2, Target, ChevronRight, Lightbulb, LayoutList
} from 'lucide-react';
import tippy from 'tippy.js';
import 'tippy.js/dist/tippy.css';
import 'tippy.js/animations/shift-away.css';

// ============================================================================
// 1. HELPER COMPONENTS (Mention Menu)
// ============================================================================

const CategorizedMentionList = forwardRef((props, ref) => {
    const { cv, extraSuggestions, query, command } = props;
    const [activeCategory, setActiveCategory] = useState(null);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
    const [mobileFilter, setMobileFilter] = useState(null);

    useEffect(() => {
        const handleResize = () => setIsMobile(window.innerWidth < 768);
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const categories = useMemo(() => {
        const safeCV = cv || {};
        const list = [];

        if (extraSuggestions && extraSuggestions.length > 0) {
            list.push({
                id: 'map',
                label: 'Evidence',
                icon: Lightbulb,
                items: extraSuggestions.map(s => ({
                    id: s.id,
                    label: s.evidence || s.label || 'Evidence',
                    detail: s.requirement || s.subtitle,
                    context: 'Evidence'
                }))
            });
        }

        if (safeCV.experiences?.length) list.push({ id: 'exp', label: 'Experience', icon: Briefcase, items: safeCV.experiences });
        if (safeCV.projects?.length) list.push({ id: 'proj', label: 'Projects', icon: FolderGit2, items: safeCV.projects });
        if (safeCV.education?.length) list.push({ id: 'edu', label: 'Education', icon: GraduationCap, items: safeCV.education });
        if (safeCV.skills?.length) list.push({ id: 'skills', label: 'Skills', icon: Cpu, items: safeCV.skills });
        if (safeCV.achievements?.length) list.push({ id: 'ach', label: 'Achievements', icon: Trophy, items: safeCV.achievements });
        if (safeCV.hobbies?.length) list.push({ id: 'hobbies', label: 'Hobbies', icon: Heart, items: safeCV.hobbies });

        return list;
    }, [cv, extraSuggestions]);

    const filteredCategories = useMemo(() => {
        const lowerQuery = query?.toLowerCase() || '';
        let baseCats = categories;

        if (isMobile && mobileFilter && !lowerQuery) {
            baseCats = categories.filter(c => c.id === mobileFilter);
        }

        if (lowerQuery) {
            return baseCats.map(cat => ({
                ...cat,
                items: cat.items.filter(item => {
                    const label = item.label || item.name || item.title || item.text || item.degree || '';
                    const detail = item.detail || item.company || item.institution || item.description || item.context || '';
                    return label.toLowerCase().includes(lowerQuery) || detail.toLowerCase().includes(lowerQuery);
                })
            })).filter(cat => cat.items.length > 0);
        }

        return baseCats;
    }, [categories, query, isMobile, mobileFilter]);

    const flatItems = useMemo(() => {
        if (query || isMobile) {
            return filteredCategories.flatMap(cat => cat.items);
        }
        const active = categories.find(c => c.id === activeCategory);
        return active ? active.items : [];
    }, [filteredCategories, categories, activeCategory, query, isMobile]);

    useEffect(() => {
        if (!isMobile && !query && categories.length > 0 && !activeCategory) {
            setActiveCategory(categories[0].id);
        }
    }, [categories, query, activeCategory, isMobile]);

    useImperativeHandle(ref, () => ({
        onKeyDown: ({ event }) => {
            if (event.key === 'ArrowUp') {
                setSelectedIndex((prev) => (prev + flatItems.length - 1) % flatItems.length);
                return true;
            }
            if (event.key === 'ArrowDown') {
                setSelectedIndex((prev) => (prev + 1) % flatItems.length);
                return true;
            }
            if (event.key === 'Enter') {
                if (flatItems[selectedIndex]) {
                    const item = flatItems[selectedIndex];
                    let context = 'Item';
                    categories.forEach(c => { if (c.items.includes(item)) context = c.label; });
                    command({
                        id: item.id,
                        label: item.label || item.name || item.title,
                        context: context
                    });
                    return true;
                }
            }
            return false;
        },
    }));

    const selectItem = (item, categoryLabel) => {
        command({
            id: item.id,
            label: item.label || item.name || item.title || item.text || item.degree,
            context: categoryLabel
        });
    };

    const getItemDetail = (item) => item.detail || item.company || item.institution || item.description || item.context || item.degree || '';
    const getItemName = (item) => item.label || item.name || item.title || item.text || item.degree

    if (query || isMobile) {
        return (
            <div className="bg-white border rounded shadow-lg overflow-hidden d-flex flex-column"
                style={{
                    width: isMobile ? '100%' : '300px',
                    maxWidth: isMobile ? '90vw' : '300px',
                    maxHeight: isMobile ? '45vh' : '300px',
                    display: 'flex',
                    flexDirection: 'column'
                }}>
                {isMobile && !query && (
                    <div className="d-flex align-items-center gap-2 p-2 border-bottom bg-light overflow-x-auto hide-scrollbar flex-shrink-0">
                        <button
                            className={`btn btn-sm btn-icon rounded-circle flex-shrink-0 d-flex align-items-center justify-content-center transition-all ${!mobileFilter ? 'bg-dark text-white shadow-sm' : 'bg-white text-muted border'}`}
                            style={{ width: 32, height: 32 }}
                            onClick={() => setMobileFilter(null)}
                            title="All"
                        >
                            <LayoutList size={14} />
                        </button>
                        <div className="vr mx-1 opacity-25"></div>
                        {categories.map(cat => (
                            <button
                                key={cat.id}
                                className={`btn btn-sm btn-icon rounded-circle flex-shrink-0 d-flex align-items-center justify-content-center transition-all ${mobileFilter === cat.id ? 'bg-primary text-white shadow-sm' : 'bg-white text-muted border'}`}
                                style={{ width: 32, height: 32 }}
                                onClick={() => setMobileFilter(mobileFilter === cat.id ? null : cat.id)}
                                title={cat.label}
                            >
                                <cat.icon size={14} />
                            </button>
                        ))}
                    </div>
                )}

                <div className="custom-scroll flex-grow-1" style={{ overflowY: 'auto', minHeight: 0, WebkitOverflowScrolling: 'touch' }}>
                    {filteredCategories.length === 0 ? (
                        <div className="p-3 text-center text-muted small">No matches found</div>
                    ) : (
                        filteredCategories.map(cat => (
                            <div key={cat.id}>
                                <div className="px-3 py-1 bg-light border-bottom border-top tiny fw-bold text-uppercase text-muted mt-0 sticky-top shadow-sm" style={{ fontSize: '0.7rem', zIndex: 1, top: 0 }}>
                                    {cat.label}
                                </div>
                                {cat.items.map(item => {
                                    const idx = flatItems.indexOf(item);
                                    return (
                                        <button
                                            key={item.id}
                                            className={`w-100 text-start btn btn-sm border-0 rounded-0 px-3 py-2 d-flex flex-column ${idx === selectedIndex ? 'bg-primary text-white' : 'text-dark hover-bg-light'}`}
                                            onClick={() => selectItem(item, cat.label)}
                                        >
                                            <span className="fw-bold small text-truncate w-100">{getItemName(item)}</span>
                                            <span className={`small text-truncate w-100 ${idx === selectedIndex ? 'text-white-50' : 'text-muted opacity-75'}`}>{getItemDetail(item)}</span>
                                        </button>
                                    );
                                })}
                            </div>
                        ))
                    )}
                </div>
            </div>
        );
    }

    return (
        <div className="bg-white shadow-lg border rounded-3 overflow-hidden d-flex" style={{ height: '280px', width: '450px' }}>
            <div className="w-35 border-end bg-light overflow-auto custom-scroll d-flex flex-column" style={{ width: '35%' }}>
                {categories.map(cat => (
                    <button
                        key={cat.id}
                        className={`w-100 btn btn-sm text-start d-flex align-items-center justify-content-between px-3 py-2 border-0 rounded-0 ${activeCategory === cat.id ? 'bg-white text-primary fw-bold shadow-inset' : 'text-muted hover-bg-white'}`}
                        onMouseEnter={() => setActiveCategory(cat.id)}
                        onMouseDown={(e) => e.preventDefault()}
                    >
                        <div className="d-flex align-items-center gap-2">
                            <cat.icon size={14} /> {cat.label}
                        </div>
                        <ChevronRight size={12} className="opacity-50" />
                    </button>
                ))}
            </div>
            <div className="w-65 overflow-auto custom-scroll bg-white" style={{ width: '65%' }}>
                {!activeCategory ? (
                    <div className="h-100 d-flex align-items-center justify-content-center text-muted small fst-italic p-3 text-center">
                        Hover a category...
                    </div>
                ) : (
                    categories.find(c => c.id === activeCategory)?.items.map((item, i) => (
                        <button
                            key={item.id || i}
                            className="w-100 btn btn-sm text-start px-3 py-2 border-bottom border-light hover-bg-primary-subtle"
                            onClick={() => selectItem(item, categories.find(c => c.id === activeCategory).label)}
                            onMouseDown={(e) => e.preventDefault()}
                        >
                            <div className="d-flex flex-column">
                                <span className="fw-bold text-dark small text-truncate">
                                    {getItemName(item)}
                                </span>
                                {getItemDetail(item) && (
                                    <span className="text-muted text-truncate" style={{ fontSize: '0.7em' }}>
                                        {getItemDetail(item)}
                                    </span>
                                )}
                            </div>
                        </button>
                    ))
                )}
            </div>
        </div>
    );
});


// ============================================================================
// 2. CODEX PARSING UTILITIES
// ============================================================================

const CODEX_REGEX = /\[([a-zA-Z0-9_]+)\]<:([a-zA-Z0-9_]+)><(.*?)>/g;

const codexToHtml = (text) => {
    if (!text) return '';
    let html = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    html = html.replace(CODEX_REGEX, (match, context, id, label) => {
        return `<span data-mention data-id="${id}" data-label="${label}" data-context="${context}" class="mention-chip">@${label}</span>`;
    });
    html = html.replace(/\n/g, '<br>');
    return html;
};

const htmlToCodex = (html) => {
    if (!html) return '';
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    doc.querySelectorAll('[data-mention]').forEach(node => {
        const id = node.getAttribute('data-id');
        const label = node.getAttribute('data-label');
        const context = node.getAttribute('data-context');
        if (id && label && context) {
            node.replaceWith(`[${context}]<:${id}><${label}>`);
        } else {
            node.replaceWith(node.textContent);
        }
    });

    doc.querySelectorAll('a').forEach(node => {
        const href = node.getAttribute('href');
        const text = node.textContent;
        node.replaceWith(`[${text}](${href})`);
    });

    let text = doc.body.innerHTML;
    text = text.replace(/<p>/g, '').replace(/<\/p>/g, '\n').replace(/<br>/g, '\n');
    const textarea = document.createElement('textarea');
    textarea.innerHTML = text;
    return textarea.value;
};

// ============================================================================
// 3. EXTENSIONS
// ============================================================================

const GhostText = Mark.create({
    name: 'ghostText',
    addAttributes() { return { class: { default: 'ghost-text-mark' } } },
    parseHTML() { return [{ tag: 'span.ghost-text-mark' }] },
    renderHTML({ HTMLAttributes }) { return ['span', mergeAttributes(HTMLAttributes, { class: 'ghost-text-mark' }), 0] },
});

const SectionTitle = Node.create({
    name: 'sectionTitle',
    group: 'block',
    atom: true,
    selectable: true,
    draggable: true,
    addAttributes() { return { label: { default: 'Section Title' } }; },
    parseHTML() { return [{ tag: 'div[data-type="section-title"]' }]; },
    renderHTML({ HTMLAttributes }) {
        return ['div', mergeAttributes(HTMLAttributes, { 'data-type': 'section-title', class: 'section-title-block' }), HTMLAttributes.label];
    },
    addNodeView() {
        return ReactNodeViewRenderer(({ node }) => (
            <NodeViewWrapper className="section-title-wrapper mb-4 mt-2">
                <div className="p-2 border-bottom border-3 border-dark d-flex align-items-center bg-light-subtle">
                    <Heading1 size={24} className="text-muted me-3 opacity-50" />
                    <h2 className="m-0 fw-bold text-dark">{node.attrs.label}</h2>
                </div>
            </NodeViewWrapper>
        ));
    }
});

const GhostSectionComponent = (props) => {
    const { node, updateAttributes, deleteNode } = props;
    const [isEditing, setIsEditing] = useState(false);
    const toggleVisibility = () => updateAttributes({ isVisible: !node.attrs.isVisible });

    return (
        <NodeViewWrapper className="ghost-section-wrapper my-4">
            <div className={`ghost-header d-flex align-items-center gap-2 p-2 rounded-top border border-dashed
        ${node.attrs.linkedId ? 'border-primary bg-primary-subtle text-primary' : 'border-purple bg-purple-subtle text-purple'}
        ${node.attrs.isVisible ? 'border-bottom-0 border-solid shadow-sm' : ''}
        `}>
                <button className="btn btn-sm btn-link p-0 text-inherit hover-opacity-100 opacity-75" onClick={toggleVisibility}>
                    {node.attrs.isVisible ? <Eye size={14} /> : <EyeOff size={14} />}
                </button>
                {isEditing ? (
                    <input autoFocus className="form-control form-control-sm border-0 bg-transparent p-0 fw-bold text-inherit w-100 shadow-none"
                        value={node.attrs.label}
                        onChange={e => updateAttributes({ label: e.target.value })}
                        onBlur={() => setIsEditing(false)}
                        onKeyDown={e => { if (e.key === 'Enter') setIsEditing(false); }}
                    />
                ) : (
                    <span className="fw-bold small flex-grow-1 cursor-text" onClick={() => setIsEditing(true)}>{node.attrs.label || "Untitled Section"}</span>
                )}
                <button className="btn btn-sm btn-link p-0 opacity-50 hover-opacity-100 text-inherit" onClick={deleteNode}><Trash2 size={14} /></button>
            </div>
            <NodeViewContent className={`ghost-content content-div p-3 border-start border-end border-bottom rounded-bottom bg-white
        ${node.attrs.linkedId ? 'border-primary-light' : 'border-purple-light'}
        ${node.attrs.isVisible ? 'border-solid' : 'border-dashed'}
        `} />
            <style>{`
        .border-purple { border-color: #8b5cf6 !important; }
        .bg-purple-subtle { background-color: #f5f3ff; color: #7c3aed; }
        .border-purple-light { border-color: rgba(139, 92, 246, 0.3) !important; }
        .border-primary-light { border-color: rgba(13, 110, 253, 0.3) !important; }
        .text-inherit { color: inherit; }
        .border-solid { border-style: solid !important; }
        .border-dashed { border-style: dashed !important; }
        `}</style>
        </NodeViewWrapper>
    );
};

const GhostSection = Node.create({
    name: 'ghostSection',
    group: 'block',
    content: '(paragraph | blockquote | bulletList | orderedList | ghostSection)+',
    defining: true,
    addAttributes() {
        return {
            label: { default: 'New Strategy Section' },
            linkedId: { default: null },
            linkedLabel: { default: null },
            linkedType: { default: null },
            isVisible: { default: false },
        };
    },
    parseHTML() { return [{ tag: 'div[data-type="ghost-section"]' }]; },
    renderHTML({ HTMLAttributes }) {
        return ['div', mergeAttributes(HTMLAttributes, { 'data-type': 'ghost-section', class: 'ghost-section-block' }), 0];
    },
    addNodeView() { return ReactNodeViewRenderer(GhostSectionComponent); },
    addInputRules() {
        return [
            new InputRule({
                find: /^>>>\s$/,
                handler: ({ state, range }) => {
                    const { tr } = state;
                    tr.replaceWith(range.from - 1, range.to, this.type.create({}, state.schema.nodes.paragraph.create()));
                },
            }),
        ];
    },
});

const MentionComponent = ({ node }) => {
    const { label, context, isGhost } = node.attrs;
    return (
        <NodeViewWrapper as="span" className={`mention-chip-wrapper ${isGhost ? 'ghost-mode' : ''}`}>
            <span className={`mention-chip ${isGhost ? 'ghost' : ''}`} title={`${context || 'Reference'}: ${label}`}>
                {isGhost && <Ghost size={10} className="me-1 opacity-50" />}
                {label}
            </span>
            <style>{`
        .mention-chip { background-color: #e7f1ff; color: #0d6efd; border-radius: 6px; padding: 2px 6px; margin: 0 2px; font-size: 0.9em; font-weight: 600; border: 1px solid rgba(13, 110, 253, 0.1); display: inline-flex; align-items: center; vertical-align: baseline; user-select: none; }
        .mention-chip.ghost { background-color: transparent; color: #6c757d; border: 1px dashed #ced4da; }
        `}</style>
        </NodeViewWrapper>
    );
};

// ============================================================================
// 4. MAIN COMPONENT
// ============================================================================


const IntelligentTextArea = forwardRef(({ initialValue, onSave, placeholder, minHeight = '150px', cv, extraSuggestions = [], onMention }, ref) => {
    const editor = useEditor({
        extensions: [
            StarterKit.configure({ heading: { levels: [1, 2, 3, 4] } }),
            Placeholder.configure({ placeholder: placeholder || 'Type @ to link evidence, >>> for sections...' }),
            Link.configure({ openOnClick: true }),
            GhostText,
            GhostSection,
            SectionTitle,
            Mention.configure({
                HTMLAttributes: { class: 'mention-chip' },
                renderText({ options, node }) { return `${node.attrs.label ?? node.attrs.id}`; },
                suggestion: {
                    items: ({ query }) => ['trigger'],
                    command: ({ editor, range, props }) => {
                        editor.chain().focus().insertContentAt(range, [
                            { type: 'mention', attrs: props },
                            { type: 'text', text: ' ' }
                        ]).run();

                        if (onMention) {
                            const typeForCallback = props.context === 'Evidence' ? 'evidence' : props.context;
                            onMention(props, typeForCallback);
                        }
                    },
                    render: () => {
                        let component;
                        let popup;
                        return {
                            onStart: props => {
                                component = new ReactRenderer(CategorizedMentionList, {
                                    props: { ...props, cv, extraSuggestions },
                                    editor: props.editor,
                                });
                                if (!props.clientRect) return;
                                popup = tippy('body', {
                                    getReferenceClientRect: props.clientRect,
                                    appendTo: () => document.body,
                                    content: component.element,
                                    showOnCreate: true,
                                    interactive: true,
                                    trigger: 'manual',
                                    placement: 'bottom-start',
                                    maxWidth: 'none',
                                    zIndex: 10000,
                                });
                            },
                            onUpdate: props => {
                                component.updateProps({ ...props, cv, extraSuggestions });
                                if (!props.clientRect) return;
                                popup[0].setProps({ getReferenceClientRect: props.clientRect });
                            },
                            onKeyDown: props => {
                                if (props.event.key === 'Escape') {
                                    popup[0].hide();
                                    return true;
                                }
                                return component.ref?.onKeyDown(props);
                            },
                            onExit: () => {
                                popup[0].destroy();
                                component.destroy();
                            },
                        };
                    },
                },
            }).extend({
                addAttributes() {
                    return {
                        id: { default: null },
                        label: { default: null },
                        context: { default: null, parseHTML: el => el.getAttribute('data-context') },
                        isGhost: { default: false, parseHTML: element => element.getAttribute('data-ghost') === 'true' },
                    };
                },
                addNodeView() { return ReactNodeViewRenderer(MentionComponent); },
            }),
        ],
        // --- CRITICAL FIX: INTERCEPT ANDROID BACKSPACE ---
        editorProps: {
            attributes: {
                class: 'ProseMirror form-control',
                style: `min-height: ${minHeight}; border: none; outline: none; box-shadow: none;`,
            },
            handleKeyDown: (view, event) => {
                if (event.key === 'Backspace') {
                    console.log('Backspace detected - checking for mention deletion');
                    const { state, dispatch } = view;
                    const { selection } = state;
                    const { $from, empty } = selection;


                    if (!empty) return false;


                    const nodeBefore = $from.nodeBefore;
                    if (nodeBefore && nodeBefore.type.name === 'mention') {
                        console.log('Deleting entire mention node');
                        const tr = state.tr.delete($from.pos - nodeBefore.nodeSize, $from.pos);
                        dispatch(tr);
                        event.preventDefault();
                        return true;
                    }
                }


                return false;
            },
            handleDOMEvents: {
                beforeinput: (view, event) => {
                    // This event is specifically triggered by Android keyboards for "Backspace"
                    if (event.inputType === 'deleteContentBackward') {
                        const { state, dispatch } = view;
                        const { selection } = state;
                        const { $from, empty } = selection;
                        // We only want to interfere if selection is collapsed (cursor)
                        if (!empty) return false;
                        // Check if the node directly BEFORE the cursor is a mention
                        const nodeBefore = $from.nodeBefore;
                        if (nodeBefore && nodeBefore.type.name === 'mention') {
                            // Manually perform atomic deletion
                            // Deletes from (cursor - nodeSize) to (cursor)
                            const tr = state.tr.delete($from.pos - nodeBefore.nodeSize, $from.pos);
                            dispatch(tr);
                            // Prevent default browser behavior (which is to try and "select" the node first)
                            event.preventDefault();
                            return true;
                        }
                    }
                    return false;
                }
            }
        },
        content: codexToHtml(initialValue),
        onBlur: ({ editor }) => {
            if (onSave) onSave(htmlToCodex(editor.getHTML()));
        },
    });

    useImperativeHandle(ref, () => ({
        focus: () => editor?.commands.focus(),
        getEditor: () => editor,
        openMenu: () => {
            if (editor) {
                editor.commands.focus('end');
                editor.commands.insertContent('@');
            }
        }
    }));

    useEffect(() => {
        if (editor && initialValue !== undefined) {
            const currentCodex = htmlToCodex(editor.getHTML());
            if (initialValue !== currentCodex) {
                editor.commands.setContent(codexToHtml(initialValue));
            }
        }
    }, [initialValue, editor]);

    if (!editor) return null;

    return (
        <div className="intelligent-textarea-container border rounded bg-white focus-within-shadow">
            <EditorContent editor={editor} className="p-2" />
            <style>{`
        .ProseMirror { outline: none; }
        .ProseMirror p.is-editor-empty:first-child::before { color: #adb5bd; content: attr(data-placeholder); float: left; height: 0; pointer-events: none; }
        .focus-within-shadow:focus-within { border-color: #86b7fe !important; box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25); }
        .ghost-text-mark { color: #6c757d; background-color: #f8f9fa; border-bottom: 2px dashed #ced4da; opacity: 0.8; }
        .hover-bg-light:hover { background-color: #f8f9fa; }
        .hover-bg-primary-subtle:hover { background-color: #cfe2ff !important; }
        .shadow-inset { box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1); }
        .sticky-top { position: sticky; top: 0; }
        /* Hide Scrollbar but allow scroll */
        .hide-scrollbar::-webkit-scrollbar { display: none; }
        .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
        `}</style>
        </div>
    );
});

export default IntelligentTextArea;