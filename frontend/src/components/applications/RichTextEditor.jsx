// frontend/src/components/applications/RichTextEditor.jsx
import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { useEditor, EditorContent, ReactRenderer } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';
import Mention from '@tiptap/extension-mention';
import Link from '@tiptap/extension-link';
import { 
    Bold, Italic, List, ListOrdered, Quote, 
    Briefcase, GraduationCap, Cpu, Heart, Trophy, 
    Lightbulb, ChevronDown, Type, Sparkles, Info 
} from 'lucide-react';
import tippy from 'tippy.js';
import 'tippy.js/dist/tippy.css';
import 'tippy.js/animations/shift-away.css';
import 'tippy.js/themes/light-border.css';

// --- 1. PORTAL DROPDOWN ---
const PortalDropdown = ({ isOpen, onClose, rect, children }) => {
    if (!isOpen || !rect) return null;
    const style = {
        position: 'fixed',
        top: `${rect.bottom + 8}px`,
        left: `${rect.left}px`,
        zIndex: 9999,
        minWidth: '240px',
        maxWidth: '300px',
        maxHeight: '300px',
        overflowY: 'auto'
    };
    return createPortal(
        <>
            <div style={{position: 'fixed', inset: 0, zIndex: 9998}} onClick={onClose} />
            <div className="bg-white border rounded-3 shadow-lg animate-fade-in" style={style}>{children}</div>
        </>,
        document.body
    );
};

// --- 2. PREVIEW TOOLTIP LOGIC ---
// We attach this to the editor's DOM after render
const attachChipPreviews = (editor) => {
    if (!editor) return;
    // Find all chips
    const chips = editor.view.dom.querySelectorAll('.mention-chip');
    chips.forEach(chip => {
        if (chip._tippy) return; // Already attached
        const label = chip.getAttribute('data-label');
        
        tippy(chip, {
            content: `
                <div class="p-2">
                    <div class="fw-bold small mb-1">${label}</div>
                    <div class="tiny text-muted">Click to edit or remove</div>
                </div>
            `,
            allowHTML: true,
            interactive: true,
            theme: 'light-border',
            animation: 'shift-away',
            delay: [200, 0]
        });
    });
};


// --- 3. MENTION LIST ---
const MentionList = React.forwardRef((props, ref) => {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const selectItem = index => { const item = props.items[index]; if (item) props.command({ id: item.id, label: item.label }); };
  useEffect(() => setSelectedIndex(0), [props.items]);
  React.useImperativeHandle(ref, () => ({
    onKeyDown: ({ event }) => {
      if (event.key === 'ArrowUp') { setSelectedIndex((selectedIndex + props.items.length - 1) % props.items.length); return true; }
      if (event.key === 'ArrowDown') { setSelectedIndex((selectedIndex + 1) % props.items.length); return true; }
      if (event.key === 'Enter') { selectItem(selectedIndex); return true; }
      return false;
    },
  }));
  return (
    <div className="bg-white border rounded shadow-lg overflow-hidden" style={{minWidth: '200px', maxWidth: '300px'}}>
      {props.items.length ? props.items.map((item, index) => (
        <button key={index} className={`w-100 text-start btn btn-sm border-0 rounded-0 px-3 py-2 d-flex flex-column ${index === selectedIndex ? 'bg-primary text-white' : 'text-dark hover-bg-light'}`} onClick={() => selectItem(index)}>
           <span className="fw-bold small text-truncate w-100">{item.label}</span>
           {item.context && <span className={`small text-truncate w-100 ${index === selectedIndex ? 'text-white-50' : 'text-muted'}`} style={{fontSize: '0.7rem'}}>{item.context}</span>}
        </button>
      )) : <div className="p-2 text-muted small">No matches found</div>}
    </div>
  );
});

// --- 4. TOOLBAR DROPDOWN ---
const ToolbarDropdown = ({ icon: Icon, label, tooltip, items, colorClass = "text-muted", onInsert }) => {
    const [isOpen, setIsOpen] = useState(false);
    const buttonRef = useRef(null);
    const [rect, setRect] = useState(null);

    useEffect(() => {
        if (buttonRef.current) {
            tippy(buttonRef.current, { content: tooltip || label, placement: 'top', animation: 'shift-away', theme: 'light-border' });
        }
    }, [tooltip, label]);

    const hasStrategyItems = items.some(i => i.isStrategy);

    return (
        <>
            <button 
                ref={buttonRef}
                onClick={() => {
                    if (buttonRef.current) {
                        setRect(buttonRef.current.getBoundingClientRect());
                        setIsOpen(!isOpen);
                    }
                }}
                className={`btn btn-sm d-flex align-items-center gap-2 px-2 py-1 transition-all position-relative ${isOpen ? 'bg-light-subtle shadow-inset' : 'hover-bg-white'}`}
                style={{border: '1px solid transparent', borderRadius: '6px', height: '32px', whiteSpace: 'nowrap'}}
            >
                <Icon size={14} className={colorClass} />
                <span className={`small fw-bold d-none d-xl-block ${colorClass}`}>{label}</span>
                {hasStrategyItems && <span className="position-absolute top-0 start-100 translate-middle p-1 bg-primary border border-light rounded-circle" style={{width: 8, height: 8}}></span>}
                <ChevronDown size={10} className="text-muted opacity-50"/>
            </button>

            <PortalDropdown isOpen={isOpen} rect={rect} onClose={() => setIsOpen(false)}>
                <div className="px-2 py-2 bg-light border-bottom sticky-top">
                    <span className="tiny fw-bold text-uppercase text-muted" style={{fontSize: '0.65rem', letterSpacing: '0.5px'}}>
                        Insert {tooltip || label}
                    </span>
                </div>
                {items.length === 0 ? (
                    <div className="p-3 text-center small text-muted fst-italic">No items found.</div>
                ) : (
                    items.map((item, idx) => (
                        <button 
                            key={idx} 
                            className="w-100 btn btn-sm text-start text-truncate hover-bg-light small d-flex align-items-center justify-content-between py-2 px-3 border-bottom border-light"
                            onClick={() => { onInsert(item); setIsOpen(false); }}
                        >
                            <div className="d-flex flex-column overflow-hidden me-2">
                                <span className="text-dark fw-medium text-truncate">{item.label}</span>
                                {item.detail && <span className="text-muted text-truncate" style={{fontSize: '0.7em'}}>{item.detail}</span>}
                            </div>
                            {item.isStrategy && (
                                <div className="bg-primary-subtle text-primary rounded-circle p-1" title="In Strategy">
                                    <div style={{width: 6, height: 6, borderRadius: '50%', backgroundColor: 'currentColor'}} />
                                </div>
                            )}
                        </button>
                    ))
                )}
            </PortalDropdown>
        </>
    );
};

// --- 5. MAIN EDITOR ---
const RichTextEditor = ({ initialContent, onUpdate, placeholder, strategyArgs = [], cvCategories = {}, hints = [] }) => {
    const allMentionItems = React.useMemo(() => {
        const combined = [];
        strategyArgs.forEach(arg => {
            combined.push({ id: `arg-${arg.id}`, label: arg.title, context: 'Strategy Argument' });
            arg.evidence.forEach(ev => combined.push({ id: ev.id, label: ev.label, context: `Evidence: ${arg.title}` }));
        });
        Object.entries(cvCategories).forEach(([cat, items]) => {
            items.forEach(i => combined.push({ id: i.id, label: i.label, context: cat })); 
        });
        return combined;
    }, [strategyArgs, cvCategories]);

    const editor = useEditor({
        extensions: [
            StarterKit,
            Placeholder.configure({ placeholder: placeholder || 'Write your cover letter...' }),
            Link.configure({ openOnClick: false }),
            Mention.configure({
                HTMLAttributes: { class: 'mention-chip', 'data-label': 'Mention' }, // data-label updated on render
                renderLabel({ options, node }) { return `${node.attrs.label ?? node.attrs.id}`; },
                suggestion: {
                    items: ({ query }) => allMentionItems.filter(i => i.label.toLowerCase().includes(query.toLowerCase()) || i.context.toLowerCase().includes(query.toLowerCase())).slice(0, 10),
                    render: () => {
                        let component, popup;
                        return {
                            onStart: props => {
                                component = new ReactRenderer(MentionList, { props, editor: props.editor });
                                if (!props.clientRect) return;
                                popup = tippy('body', {
                                    getReferenceClientRect: props.clientRect, appendTo: () => document.body,
                                    content: component.element, showOnCreate: true, interactive: true,
                                    trigger: 'manual', placement: 'bottom-start',
                                });
                            },
                            onUpdate: props => { component.updateProps(props); if (!props.clientRect) return; popup[0].setProps({ getReferenceClientRect: props.clientRect }); },
                            onKeyDown: props => { if (props.event.key === 'Escape') { popup[0].hide(); return true; } return component.ref?.onKeyDown(props); },
                            onExit: () => { popup[0].destroy(); component.destroy(); },
                        };
                    },
                },
            }),
        ],
        content: initialContent,
        onUpdate: ({ editor }) => {
            // Add data-label attributes to chips for the tooltip to read
            editor.view.dom.querySelectorAll('.mention-chip').forEach(el => {
                // Tiptap puts label in textContent, we just sync it for valid HTML parsing if needed
                el.setAttribute('data-label', el.textContent); 
            });
            attachChipPreviews(editor); // Refresh tooltips
        },
        onBlur: ({ editor }) => onUpdate(editor.getHTML()),
    });

    // Initial tooltip attachment
    useEffect(() => { if(editor) attachChipPreviews(editor); }, [editor, initialContent]);

    const insertChip = (item) => editor?.chain().focus().insertContent({ type: 'mention', attrs: { id: item.id, label: item.label } }).insertContent(' ').run();

    const ToolbarDivider = () => <div className="vr mx-2 opacity-25 h-50 my-auto flex-shrink-0"></div>;
    const ToolbarBtn = ({ action, icon: Icon, isActive, tooltip }) => {
        const btnRef = useRef(null);
        useEffect(() => { if(btnRef.current) tippy(btnRef.current, { content: tooltip, theme: 'light-border', animation: 'shift-away' }); }, [tooltip]);
        return <button ref={btnRef} onClick={action} className={`btn btn-sm border-0 d-flex align-items-center justify-content-center flex-shrink-0 transition-all ${isActive ? 'text-primary bg-primary-subtle' : 'text-muted hover-text-dark hover-bg-light'}`} style={{width: '32px', height: '32px', borderRadius: '6px'}} type="button"><Icon size={16} /></button>;
    };

    if (!editor) return null;

    return (
        <div className="d-flex flex-column h-100 border rounded-3 bg-white focus-within-shadow transition-all" style={{minHeight: '400px'}}>
            
            {/* --- SMART HINT BAR --- */}
            {hints.length > 0 && (
                <div className="bg-blue-50 border-bottom px-3 py-2 d-flex align-items-start gap-2 animate-fade-in">
                    <Sparkles size={14} className="text-primary mt-1 flex-shrink-0" />
                    <div className="small text-dark flex-grow-1">
                        <span className="fw-bold text-primary me-1">AI Tip:</span> 
                        {hints[0]}
                    </div>
                    <button className="btn-close small" style={{fontSize: '0.5rem'}} onClick={() => {/* Dismiss logic if needed */}}></button>
                </div>
            )}

            {/* --- TOP TOOLBAR (Scrollable on Mobile) --- */}
            <div className="d-flex align-items-center gap-2 p-2 border-bottom bg-light-subtle rounded-top-3 overflow-x-auto hide-scrollbar text-nowrap">
                {strategyArgs.length > 0 && (
                    <div className="d-flex gap-2 align-items-center">
                        {strategyArgs.map(arg => (
                            <ToolbarDropdown key={arg.id} icon={Lightbulb} label={arg.title.length > 15 ? arg.title.substring(0, 15) + '...' : arg.title} tooltip={`Strategy: ${arg.title}`} items={arg.evidence} colorClass="text-primary" onInsert={insertChip} />
                        ))}
                        <ToolbarDivider />
                    </div>
                )}
                <ToolbarDropdown icon={Briefcase} label="Exp" tooltip="Experience" items={cvCategories.Experience || []} onInsert={insertChip} />
                <ToolbarDropdown icon={GraduationCap} label="Edu" tooltip="Education" items={cvCategories.Education || []} onInsert={insertChip} />
                <ToolbarDropdown icon={Cpu} label="Skills" tooltip="Skills" items={cvCategories.Skills || []} onInsert={insertChip} />
                <ToolbarDropdown icon={Trophy} label="Achieve" tooltip="Achievements" items={cvCategories.Achievements || []} onInsert={insertChip} />
                <ToolbarDropdown icon={Heart} label="Hobbies" tooltip="Hobbies" items={cvCategories.Hobbies || []} onInsert={insertChip} />
            </div>

            {/* --- BOTTOM TOOLBAR --- */}
            <div className="d-flex align-items-center gap-1 px-2 py-1 border-bottom bg-white overflow-x-auto hide-scrollbar">
                <ToolbarBtn icon={Bold} action={() => editor.chain().focus().toggleBold().run()} isActive={editor.isActive('bold')} tooltip="Bold" />
                <ToolbarBtn icon={Italic} action={() => editor.chain().focus().toggleItalic().run()} isActive={editor.isActive('italic')} tooltip="Italic" />
                <ToolbarDivider />
                <ToolbarBtn icon={List} action={() => editor.chain().focus().toggleBulletList().run()} isActive={editor.isActive('bulletList')} tooltip="Bullet List" />
                <ToolbarBtn icon={ListOrdered} action={() => editor.chain().focus().toggleOrderedList().run()} isActive={editor.isActive('orderedList')} tooltip="Ordered List" />
                <ToolbarBtn icon={Quote} action={() => editor.chain().focus().toggleBlockquote().run()} isActive={editor.isActive('blockquote')} tooltip="Blockquote" />
                <div className="ms-auto text-muted small d-flex align-items-center gap-2 pe-2 flex-shrink-0" style={{fontSize: '0.7rem'}}>
                    <Type size={12}/> {editor.storage.characterCount?.characters?.() || 0}
                </div>
            </div>
            
            {/* --- EDITOR CANVAS --- */}
            <EditorContent editor={editor} className="flex-grow-1 p-4" />
            
            <style>{`
                .ProseMirror { outline: none; min-height: 100%; font-size: 1rem; line-height: 1.7; color: #333; }
                .ProseMirror p.is-editor-empty:first-child::before { color: #adb5bd; content: attr(data-placeholder); float: left; height: 0; pointer-events: none; }
                .mention-chip { background-color: #e7f1ff; color: #0d6efd; border-radius: 20px; padding: 2px 8px; margin: 0 2px; font-size: 0.9em; font-weight: 600; border: 1px solid rgba(13, 110, 253, 0.1); cursor: default; display: inline-block; vertical-align: middle; }
                .mention-chip:hover { background-color: #d0e1fd; border-color: #86b7fe; }
                .focus-within-shadow:focus-within { box-shadow: 0 0 0 4px rgba(13, 110, 253, 0.1); border-color: #86b7fe !important; }
                .hide-scrollbar::-webkit-scrollbar { display: none; } .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
                .bg-blue-50 { background-color: #f0f7ff; }
            `}</style>
        </div>
    );
};

export default RichTextEditor;