"""Safe Assistant panel (V4-M12), an Analysis-page tab.

A bounded, chat-like helper. Suggested-action buttons and a free-text box
both route only to the deterministic assistant engine (assistant.py):
the box goes through interpret_request, which maps to a safe action or
refuses. The assistant organizes computed evidence -- it never asserts a
physical cause. main_window supplies the computed context (session,
notebook, current view) and runs the engine; this panel is thin UI.

Outputs are savable to the Evidence Notebook (V4-M2).
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from assistant import DISCLAIMER


_ACTION_BUTTONS = (
    ("Summarize session", "summarize_session"),
    ("List key findings", "list_key_findings"),
    ("Report outline", "report_outline"),
    ("Compare intervals", "compare_intervals"),
    ("Figure caption", "figure_caption"),
)


class AssistantPanel(QtWidgets.QWidget):
    action_requested = QtCore.pyqtSignal(str)   # a SAFE_ACTIONS name
    query_submitted = QtCore.pyqtSignal(str)    # free text -> interpret_request
    save_requested = QtCore.pyqtSignal(str)     # save the last output to the notebook

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_output = ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Assistant")
        title.setProperty("role", "section-title")
        layout.addWidget(title)

        self.caption = QtWidgets.QLabel(
            "A bounded helper: it organizes your computed evidence into summaries, "
            "finding lists, outlines, comparisons, and captions. It never infers a "
            "physical cause -- \"why\" questions are declined by design.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        actions = QtWidgets.QHBoxLayout()
        for text, action in _ACTION_BUTTONS:
            btn = QtWidgets.QPushButton(text)
            btn.setAccessibleName(f"assistant-{action}")
            btn.clicked.connect(lambda _c=False, a=action: self.action_requested.emit(a))
            actions.addWidget(btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.output = QtWidgets.QTextEdit()
        self.output.setAccessibleName("Assistant output")
        self.output.setReadOnly(True)
        layout.addWidget(self.output, 1)

        input_row = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("Ask for a summary, findings, outline, comparison, or caption…")
        self.input.setAccessibleName("Assistant input")
        self.input.returnPressed.connect(self._submit)
        send = QtWidgets.QPushButton("Send")
        send.setAccessibleName("assistant-send")
        send.clicked.connect(self._submit)
        self.save_button = QtWidgets.QPushButton("Save to notebook")
        self.save_button.setAccessibleName("assistant-save")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(lambda: self.save_requested.emit(self._last_output))
        input_row.addWidget(self.input, 1)
        input_row.addWidget(send)
        input_row.addWidget(self.save_button)
        layout.addLayout(input_row)

    def _submit(self) -> None:
        text = self.input.text().strip()
        if text:
            self.query_submitted.emit(text)
            self.input.clear()

    def show_result(self, text: str, *, savable: bool = True) -> None:
        """Display an assistant result (appended, chat-like). Refusals are
        shown but not savable."""
        self._last_output = text if savable else ""
        self.output.append(text + "\n")
        self.save_button.setEnabled(savable and bool(text))

    @property
    def last_output(self) -> str:
        return self._last_output
