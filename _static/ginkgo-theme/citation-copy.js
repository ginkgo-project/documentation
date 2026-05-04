// Copy-to-clipboard for the homepage citation block.
document.addEventListener("DOMContentLoaded", () => {
    const buttons = document.querySelectorAll(".ginkgo-cite-copy");
    buttons.forEach((btn) => {
        btn.addEventListener("click", async () => {
            const targetSel = btn.getAttribute("data-target");
            const target = targetSel ? document.querySelector(targetSel) : null;
            if (!target) return;
            const text = target.textContent.trim();
            try {
                await navigator.clipboard.writeText(text);
            } catch (err) {
                // Fallback for older browsers / non-secure contexts.
                const ta = document.createElement("textarea");
                ta.value = text;
                ta.setAttribute("readonly", "");
                ta.style.position = "absolute";
                ta.style.left = "-9999px";
                document.body.appendChild(ta);
                ta.select();
                try { document.execCommand("copy"); } catch (_) {}
                document.body.removeChild(ta);
            }
            const originalLabel = btn.textContent;
            btn.textContent = "Copied!";
            btn.classList.add("is-copied");
            setTimeout(() => {
                btn.textContent = originalLabel;
                btn.classList.remove("is-copied");
            }, 1500);
        });
    });
});
