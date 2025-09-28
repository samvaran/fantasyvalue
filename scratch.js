[...document.body.querySelectorAll(".component-18-wrapper")].map((game) => {
  const players = [
    ...game.querySelectorAll(".component-18-side-rail > .side-rail-name"),
  ]
    .map((p) => p.innerText)
    .filter((p) => !p.toLocaleLowerCase().includes("no touchdown"));

  const anytimeTdOddsParent = [
    ...game.querySelectorAll(".component-18 > .participants"),
  ].find((c) => {
    return c.innerText.toLocaleLowerCase().includes("anytime td");
  }).parentNode;
  const anytimeTdOdds = [
    ...anytimeTdOddsParent.querySelectorAll(
      ".outcomes > .component-18__cell span.sportsbook-odds"
    ),
  ].map((o) => o.innerText);

  return anytimeTdOdds;
});
