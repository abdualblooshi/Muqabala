class LanguageDialog {
  constructor() {
    this.dialog = document.getElementById("languageConfirmDialog");
    this.setupEventListeners();
  }

  setupEventListeners() {
    if (!this.dialog) return;

    const cancelButton = this.dialog.querySelector(".confirm-no");
    const confirmButton = this.dialog.querySelector(".confirm-yes");

    if (cancelButton) {
      cancelButton.addEventListener("click", () => this.cancelLanguageChange());
    }

    if (confirmButton) {
      confirmButton.addEventListener("click", () =>
        this.confirmLanguageChange()
      );
    }
  }

  show() {
    if (this.dialog) {
      this.dialog.classList.add("show");
    }
  }

  hide() {
    if (this.dialog) {
      this.dialog.classList.remove("show");
    }
  }

  cancelLanguageChange() {
    this.hide();
    // Reset the language selector to current language
    const langSelect = document.querySelector(
      'select[aria-label="Language / اللغة"]'
    );
    if (langSelect) {
      langSelect.value = langSelect.dataset.currentLang;
      langSelect.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }

  confirmLanguageChange() {
    // Set a flag in sessionStorage to indicate confirmed change
    sessionStorage.setItem("confirmLanguageChange", "true");
    // Submit the form to trigger the language change
    const langSelect = document.querySelector(
      'select[aria-label="Language / اللغة"]'
    );
    if (langSelect) {
      langSelect.dispatchEvent(new Event("change", { bubbles: true }));
    }
    this.hide();
  }
}

// Initialize dialog functionality when the DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  const dialog = new LanguageDialog();

  // Show dialog when language is changed during interview
  const langSelect = document.querySelector(
    'select[aria-label="Language / اللغة"]'
  );
  if (langSelect) {
    langSelect.addEventListener("change", (event) => {
      const isInterviewStarted = langSelect.dataset.interviewStarted === "true";
      if (
        isInterviewStarted &&
        event.target.value !== event.target.dataset.currentLang
      ) {
        dialog.show();
      }
    });
  }
});
