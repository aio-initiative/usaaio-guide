(function () {
  const resources = [
    {
      title: "NumPy Learn",
      type: "Docs",
      summary: "Start here for arrays, broadcasting, and dtypes. The absolute-basics page covers most of what Round 1 needs.",
      href: "https://numpy.org/learn/",
    },
    {
      title: "scikit-learn User Guide",
      type: "Docs",
      summary: "The reference for every classical model, and the place to check what a preprocessing step actually does before you use it.",
      href: "https://scikit-learn.org/stable/user_guide.html",
    },
    {
      title: "PyTorch Tutorials",
      type: "Docs",
      summary: "Official tutorials. The 60-minute blitz is enough to start; come back for the autograd mechanics when something breaks.",
      href: "https://docs.pytorch.org/tutorials/",
    },
    {
      title: "Google ML Crash Course",
      type: "Course",
      summary: "Short interactive lessons on loss, generalization, and validation. Useful if a module moved faster than you wanted.",
      href: "https://developers.google.com/machine-learning/crash-course",
    },
  ];

  const seasonPhases = [
    {
      date: "Jun 2026-Jan 2027",
      title: "Foundations",
      body: "Python and NumPy, followed by linear algebra and probability.",
    },
    {
      date: "Dec-Jan",
      title: "Round 1 practice",
      body: "Past papers and timed practice.",
    },
    {
      date: "Feb 19, 2027",
      title: "Round 1",
      body: "Proctored and onsite: a 60-minute non-coding part, then a 75-minute coding part in Google Colab.",
    },
    {
      date: "Mar-Apr 2027",
      title: "Round 2",
      body: "The full syllabus is fair game, and problems combine topics. GPU work may appear.",
    },
    {
      date: "Jun 2027",
      title: "USAAIO camp",
      body: "Selection camp with written and oral work.",
    },
  ];

  const checklistItems = [
    "Registration, eligibility, and proctoring details from usaaio.org.",
    "A Colab notebook with rendered maths and cells that run from top to bottom.",
    "At least one complete Round 1 paper under time.",
    "A short list of recurring NumPy and PyTorch shape errors.",
    "A review of the written explanations in past problems.",
  ];

  window.AIO_SITE_DATA = {
    resources,
    seasonPhases,
    checklistItems,
  };
})();
