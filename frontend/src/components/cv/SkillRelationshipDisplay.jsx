// frontend/src/components/cv/SkillRelationshipDisplay.jsx
import React from 'react';

// This component renders a single relationship entry
const RelationshipLink = ({ link }) => {
  const { title, direct, via } = link;
  const hasDirect = direct;
  const hasVia = via && via.length > 0;

  return (
    <li className="list-group-item d-flex flex-column py-2 px-0 border-0">
      <span className="fw-bold">{title}</span>
      <div className="ps-2">
        {hasDirect && (
          <span className="small text-success d-block">&bull; Linked directly</span>
        )}
        {hasVia && via.map(ach => (
          <span key={ach.id} className="small text-info d-block">
            &bull; Linked via achievement: <em>"{ach.text}"</em>
          </span>
        ))}
      </div>
    </li>
  );
};

// This component renders all the relationship groups for a skill
const SkillRelationshipDisplay = ({ relationships }) => {
  // Helper to convert map to array for rendering
  const getLinks = (map) => map ? Array.from(map.values()) : [];

  const sections = [
    { title: 'Experiences', links: getLinks(relationships.experiences), icon: 'briefcase-fill' },
    { title: 'Education', links: getLinks(relationships.education), icon: 'book-fill' },
    { title: 'Projects', links: getLinks(relationships.projects), icon: 'tools' },
    { title: 'Hobbies', links: getLinks(relationships.hobbies), icon: 'controller' },
    { title: 'Achievements', links: getLinks(relationships.achievements), icon: 'trophy-fill' },
  ];

  return (
    <div className="small">
      {sections.map(section => (
        section.links.length > 0 && (
          <div key={section.title} className="mb-2">
            <strong className="text-dark d-block">
              <i className={`bi bi-${section.icon} me-2`}></i>
              {section.title}
            </strong>
            <ul className="list-group list-group-flush small" style={{ marginLeft: '10px' }}>
              {section.links.map(link => (
                <RelationshipLink key={link.item.id} link={link} />
              ))}
            </ul>
          </div>
        )
      ))}
    </div>
  );
};

export default SkillRelationshipDisplay;