#![no_main]

//! Check that the generated TextMate grammar highlights source code consistently with the actual tokenizer.
//!
//! This uses the real TextMate engine from VS Code through a node subprocess,
//!   which requires running `npm install` in the `textmate` folder first.

use hwl_language::syntax::external::textmate::{generate_textmate_language_json, source_scope, token_scope};
use hwl_language::syntax::source::FileId;
use hwl_language::syntax::token::{Token, TokenType, tokenize};
use libfuzzer_sys::fuzz_target;
use std::fmt::Write as _;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

fuzz_target!(|data: &[u8]| target(data));

fn target(data: &[u8]) {
    let Ok(source) = std::str::from_utf8(data) else {
        return;
    };
    // editors split lines before passing them to the TextMate engine, so it never sees these characters
    if source.contains('\r') {
        return;
    }
    let Ok(tokens) = tokenize(FileId::dummy(), source, false) else {
        return;
    };

    let expected = expected_scopes(source, &tokens);
    let actual = ORACLE.lock().unwrap().tokenize(source);

    // newlines are not passed to TextMate, so they never get a scope
    let mismatch = (0..source.len()).find(|&i| source.as_bytes()[i] != b'\n' && expected[i] != actual[i]);
    if let Some(index) = mismatch {
        panic!("{}", mismatch_message(source, &expected, &actual, index));
    }
}

/// The innermost scope each byte of the source should get according to the tokenizer.
fn expected_scopes(source: &str, tokens: &[Token]) -> Vec<Option<String>> {
    let scope_source = source_scope();
    let scope_substitution = token_scope(TokenType::StringSubStart);

    // bytes that are not part of any token (eg. whitespace) get the scope of their context
    let mut result = vec![Some(scope_source.clone()); source.len()];
    let mut context = vec![Some(scope_source)];

    for (i, token) in tokens.iter().enumerate() {
        let span = token.span.start_byte..token.span.end_byte;
        result[span.clone()].fill(token_scope(token.ty));

        match token.ty {
            TokenType::StringSubStart => context.push(scope_substitution.clone()),
            TokenType::StringSubEnd => {
                context.pop();
            }
            _ => {}
        }

        // fill the gap until the next token with the current context
        let context_scope = context.last().unwrap().clone();
        let next_start = tokens.get(i + 1).map_or(source.len(), |t| t.span.start_byte);
        result[span.end..next_start].fill(context_scope);
    }

    result
}

fn mismatch_message(source: &str, expected: &[Option<String>], actual: &[Option<String>], index: usize) -> String {
    let line_start = source[..index].rfind('\n').map_or(0, |i| i + 1);
    let line_end = source[index..].find('\n').map_or(source.len(), |i| index + i);

    let mut f = String::new();
    writeln!(f, "TextMate scope mismatch at byte {index}").unwrap();
    writeln!(f, "  line:     {:?}", &source[line_start..line_end]).unwrap();
    writeln!(f, "  column:   {}", index - line_start).unwrap();
    writeln!(f, "  char:     {:?}", source[index..].chars().next().unwrap()).unwrap();
    writeln!(f, "  expected: {:?}", expected[index]).unwrap();
    writeln!(f, "  actual:   {:?}", actual[index]).unwrap();
    writeln!(f, "  source:   {source:?}").unwrap();
    f
}

static ORACLE: Mutex<Oracle> = Mutex::new(Oracle { process: None });

struct Oracle {
    process: Option<(Child, ChildStdin, BufReader<ChildStdout>)>,
}

impl Oracle {
    /// Tokenize the source with TextMate, returning the innermost scope of each byte.
    fn tokenize(&mut self, source: &str) -> Vec<Option<String>> {
        let (_, stdin, stdout) = self.process.get_or_insert_with(start_oracle);

        writeln!(stdin, "{}", serde_json::to_string(source).unwrap()).unwrap();
        stdin.flush().unwrap();
        let mut line = String::new();
        stdout.read_line(&mut line).unwrap();

        let tokens: Vec<(usize, usize, Vec<String>)> = serde_json::from_str(&line).unwrap();
        let mut result = vec![None; source.len()];
        for (start, end, scopes) in tokens {
            result[start..end].fill(scopes.last().cloned());
        }
        result
    }
}

fn start_oracle() -> (Child, ChildStdin, BufReader<ChildStdout>) {
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/textmate/oracle.mjs");
    let mut child = Command::new("node")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to start TextMate oracle, is node installed?");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = BufReader::new(child.stdout.take().unwrap());

    let grammar = generate_textmate_language_json();
    writeln!(stdin, "{}", serde_json::to_string(&grammar).unwrap()).unwrap();
    stdin.flush().unwrap();

    (child, stdin, stdout)
}
