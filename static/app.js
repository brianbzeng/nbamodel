document.addEventListener("DOMContentLoaded", () => {
  const loadingBar = document.getElementById("loading-bar");
 const forms = document.querySelectorAll("form[data-show-loading='true']");
 const themeToggle = document.getElementById("theme-toggle");
 const injuryMode = document.getElementById("injury-mode");
 const injurySubmit = document.getElementById("injury-submit");
 const injuryJobStatus = document.getElementById("injury-job-status");

 const syncInjuryMode = () => {
   if (!injuryMode) {
     return;
   }

   const selectedMode = injuryMode.value;
   document.querySelectorAll("[data-injury-fields]").forEach((group) => {
     const isActive = group.dataset.injuryFields === selectedMode;
     group.hidden = !isActive;
     group.querySelectorAll("input").forEach((input) => {
       input.disabled = !isActive;
       input.required = isActive;
     });
   });

   if (injurySubmit) {
     const labels = {
       latest: "Get latest report",
       date_range: "Scrape date range",
       season_range: "Scrape season range",
     };
     injurySubmit.textContent = labels[selectedMode] || "Scrape injury reports";
   }
 };

 if (injuryMode) {
   injuryMode.addEventListener("change", syncInjuryMode);
   syncInjuryMode();
 }

 if (injuryJobStatus) {
   const statusUrl = injuryJobStatus.dataset.statusUrl;
   const pollJob = async () => {
     try {
       const response = await fetch(statusUrl, { cache: "no-store" });
       const result = await response.json();
       if (result.status === "complete" || result.status === "error") {
         window.location.reload();
         return;
       }
     } catch (error) {
       console.warn("Unable to check injury scrape status.", error);
     }
     window.setTimeout(pollJob, 2500);
   };
   window.setTimeout(pollJob, 2500);
 }

 if (themeToggle) {
   themeToggle.addEventListener("click", () => {
     const current = document.documentElement.getAttribute("data-theme");
     const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
     const isDark = current === "dark" || (!current && prefersDark);
     const next = isDark ? "light" : "dark";
     document.documentElement.setAttribute("data-theme", next);
     localStorage.setItem("theme", next);
   });
 }
 
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
