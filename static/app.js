document.addEventListener("DOMContentLoaded", () => {
  const loadingBar = document.getElementById("loading-bar");
  const forms = document.querySelectorAll("form[data-show-loading='true']");

  const showLoading = () => {
    if (loadingBar) {
      loadingBar.classList.add("is-visible");
    }
    document.body.classList.add("is-loading");
  };

  const hideLoading = () => {
    if (loadingBar) {
      loadingBar.classList.remove("is-visible");
    }
    document.body.classList.remove("is-loading");
  };

  forms.forEach((form) => {
    form.addEventListener("submit", (event) => {
      const confirmMessage = form.getAttribute("data-confirm");
      if (confirmMessage && !window.confirm(confirmMessage)) {
        event.preventDefault();
        return;
      }

      showLoading();

      const buttons = form.querySelectorAll("button[type='submit']");
      buttons.forEach((button) => {
        if (!button.dataset.originalText) {
          button.dataset.originalText = button.textContent || "";
        }
        button.textContent = "Working...";
        button.disabled = true;
      });
    });
  });

  window.addEventListener("pageshow", hideLoading);
});
