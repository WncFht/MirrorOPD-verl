module.exports = {
  noBanner: true,
  noProgress: true,
  config: {
    MD013: false,
    MD007: { indent: 4 },
    MD024: { siblings_only: true },
    MD029: false,
    MD033: {
      allowed_elements: [
        "a",
        "br",
        "details",
        "summary",
        "img",
        "sup",
        "sub",
        "span",
        "div",
      ],
    },
    MD060: false,
    // Prettier owns emphasis layout; both legal forms occur in mathematical prose.
    MD049: false,
  },
};
