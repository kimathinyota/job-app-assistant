// frontend/src/components/common/DeleteConfirmationModal.jsx
import React, { useState, useEffect } from 'react';

const DeleteConfirmationModal = ({
  isOpen,
  onClose,
  onConfirm,
  itemName,
  itemType,
}) => {
  const [confirmationText, setConfirmationText] = useState('');
  const [isConfirmed, setIsConfirmed] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setConfirmationText('');
    }
  }, [isOpen]);

  useEffect(() => {
    setIsConfirmed(confirmationText === itemName);
  }, [confirmationText, itemName]);

  if (!isOpen) return null;

  return (
    <div
      className="modal"
      style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.5)' }}
      onClick={onClose}
    >
      <div
        className="modal-dialog modal-dialog-centered"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-content">
          <div className="modal-header">
            <h5 className="modal-title text-danger">Confirm Deletion</h5>
            <button
              type="button"
              className="btn-close"
              onClick={onClose}
            ></button>
          </div>
          <div className="modal-body">
            <p>
              This is a destructive action and cannot be undone. This will
              permanently delete the {itemType}{' '}
              <strong>{itemName}</strong>.
            </p>
            <p>
              Please type the name of the {itemType} to confirm:
            </p>
            <input
              type="text"
              className="form-control"
              value={confirmationText}
              onChange={(e) => setConfirmationText(e.target.value)}
              placeholder={itemName}
            />
          </div>
          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-danger"
              onClick={onConfirm}
              disabled={!isConfirmed}
            >
              Delete
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeleteConfirmationModal;