//! Agent skills bundled in the binary.
//!
//! Each is authored once under `skills/<name>/SKILL.md` (what `npx skills add
//! coroboros/scrybe` installs) and embedded via `include_str!`, so the binary and the
//! installable skill can't drift.

/// A bundled agent skill: its name, a one-line summary for `skills list`, and the
/// full SKILL.md body printed verbatim by `skills get`.
pub struct Skill {
    pub name: &'static str,
    pub summary: &'static str,
    pub body: &'static str,
}

/// scrybe's own usage guide for AI agents. The single source is the SKILL.md that
/// ships in the repo and installs via `npx skills add coroboros/scrybe`.
pub const SCRYBE: Skill = Skill {
    name: "scrybe",
    summary: "scrybe CLI usage guide for AI agents",
    body: include_str!("../../skills/scrybe/SKILL.md"),
};

/// Every skill the binary ships. The first is the default for `skills get`.
pub const BUNDLED: &[&Skill] = &[&SCRYBE];

/// Look up a bundled skill by exact name.
pub fn find(name: &str) -> Option<&'static Skill> {
    BUNDLED.iter().find(|skill| skill.name == name).copied()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)] // tests may expect; the binary may not
    use super::*;

    #[test]
    fn default_skill_is_findable_and_carries_its_frontmatter() {
        // `skills get` with no name resolves the first bundled skill; pin that the
        // embedded body is the real SKILL.md (its frontmatter `name:`), so a moved
        // or emptied file fails here rather than shipping an empty `get`.
        let skill = find(SCRYBE.name).expect("scrybe skill is bundled");
        assert_eq!(skill.name, BUNDLED[0].name, "default get → first bundled");
        assert!(
            skill.body.contains("name: scrybe"),
            "embedded body is not the SKILL.md frontmatter"
        );
    }

    #[test]
    fn unknown_skill_is_absent() {
        assert!(find("nope").is_none());
    }
}
