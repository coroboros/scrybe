//! The void-tinted color layer.
//!
//! Styles are defined once as [`anstyle::Style`] constants. All output goes
//! through `anstream`, which strips ANSI when stdout/stderr is not a terminal
//! or when `NO_COLOR` is set, and forces color on `CLICOLOR_FORCE=1`. [`init`]
//! adds one rule on top: the `--no-color` flag forces plain output globally.

use anstyle::{Ansi256Color, AnsiColor, Color, Effects, Style};

/// Red, bold — error lines.
pub const ERROR: Style = Style::new()
    .fg_color(Some(Color::Ansi(AnsiColor::Red)))
    .effects(Effects::BOLD);

/// Yellow — warnings and pending-feature notices.
pub const WARN: Style = Style::new().fg_color(Some(Color::Ansi(AnsiColor::Yellow)));

/// Green — success and defaults.
pub const SUCCESS: Style = Style::new().fg_color(Some(Color::Ansi(AnsiColor::Green)));

/// Dimmed — secondary detail.
pub const DIM: Style = Style::new().effects(Effects::DIMMED);

/// Void violet — the brand accent for headings.
pub const ACCENT: Style = Style::new()
    .fg_color(Some(Color::Ansi256(Ansi256Color(141))))
    .effects(Effects::BOLD);

/// Apply `--no-color`. Without it, `anstream`'s auto-detection (TTY, `NO_COLOR`,
/// `CLICOLOR_FORCE`) decides; with it, color is forced off process-wide.
pub fn init(no_color: bool) {
    if no_color {
        anstream::ColorChoice::Never.write_global();
    }
}

/// Wrap `text` in `style`'s SGR sequences. `anstream` strips them when color is
/// disabled, so callers always paint and let the stream decide.
pub fn paint(style: Style, text: &str) -> String {
    format!("{}{text}{}", style.render(), style.render_reset())
}
