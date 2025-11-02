// frontend/src/components/cv/AchievementRelationshipDisplay.jsx
import React from 'react';

// This component renders a single relationship entry
const RelationshipLink = ({ item, type }) => {
  let title = 'Unknown';
  if (type === 'experiences') title = `${item.title} @ ${item.company}`;
  if (type === 'education') title = `${item.degree} @ ${item.institution}`;
  if (type === 'projects') title = item.title;
  if (type === 'hobbies') title = item.name;

  return (
    <li className="list-group-item py-1 px-0 border-0">
      <span className="small">{title}</span>
    </li>
  );
};

// This component renders all the relationship groups for an achievement
const AchievementRelationshipDisplay = ({ relationships }) => {

  const sections = [
    { title: 'Experiences', links: relationships.experiences || [], icon: 'briefcase-fill' },
    { title: 'Education', links: relationships.education || [], icon: 'book-fill' },
    { title: 'Projects', links: relationships.projects || [], icon: 'tools' },
    { title: 'Hobbies', links: relationships.hobbies || [], icon: 'controller' },
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
              {section.links.map(item => (
                <RelationshipLink key={item.id} item={item} type={section.title.toLowerCase()} />
              ))}
            </ul>
          </div>
        )
      ))}
    </div>
  );
};

export default AchievementRelationshipDisplay;