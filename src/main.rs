#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(once_cell)]
#![feature(thread_local)]
#![feature(new_uninit)]
#![feature(arbitrary_self_types)]
#![allow(non_upper_case_globals)]

use calcium::Number;
use core::panic;
use std::{collections::{HashMap, HashSet}, hash::Hash, cell::{RefCell, Ref}};
use gc::{unsafe_empty_trace, Finalize, Gc, Trace};

macro_rules! simple_empty_finalize_trace {
    ($($T:ty),*) => {
        $(
            impl Finalize for $T {}
            unsafe impl Trace for $T { unsafe_empty_trace!(); }
        )*
    }
}

pub mod calcium {
    use std::fmt::{Debug, Display};

    use auto_ops::impl_op_ex;
    use std::cell::LazyCell;
    use std::ffi::{c_long, c_ulong, c_void, CStr, CString};

    use burrito::{
        ca_add, ca_check_equal, ca_clear, ca_ctx_clear, ca_ctx_init, ca_ctx_struct, ca_div,
        ca_get_fmpz, ca_get_str, ca_init, ca_mul, ca_one, ca_pi, ca_set_d, ca_set_fmpq, ca_set_si,
        ca_set_ui, ca_struct, ca_sub, ca_zero, flint_cleanup, flint_free, fmpq, fmpq_clear,
        fmpq_init, fmpq_set_str, fmpz, fmpz_abs_fits_ui, fmpz_clear, fmpz_get_ui, fmpz_init,
        fmpz_sgn, truth_t_T_FALSE, truth_t_T_TRUE,
    };
    use gc::{unsafe_empty_trace, Finalize, Trace};

    pub struct Context {
        ctx: *mut ca_ctx_struct,
    }
    impl Drop for Context {
        fn drop(&mut self) {
            unsafe {
                ca_ctx_clear(self.ctx);
                flint_cleanup();
                drop(Box::from_raw(self.ctx));
            }
        }
    }

    #[thread_local]
    pub static mut CALCIUM_CTX: LazyCell<Context> = LazyCell::new(|| unsafe {
        let mut ctx = Box::<ca_ctx_struct>::new_uninit();
        ca_ctx_init(ctx.as_mut_ptr());
        Context {
            ctx: Box::into_raw(ctx.assume_init()),
        }
    });

    pub struct Number {
        data: *mut ca_struct,
    }

    simple_empty_finalize_trace![Number];

    // TODO: mutable update like +=?
    impl Number {
        pub fn new() -> Number {
            unsafe {
                let mut x = Box::<ca_struct>::new_uninit();
                ca_init(x.as_mut_ptr(), CALCIUM_CTX.ctx);
                Number {
                    data: Box::into_raw(x.assume_init()),
                }
            }
        }
        pub fn zero() -> Number {
            let x = Number::new();
            unsafe {
                ca_zero(x.data, CALCIUM_CTX.ctx);
            }
            x
        }
        pub fn one() -> Number {
            let x = Number::new();
            unsafe {
                ca_one(x.data, CALCIUM_CTX.ctx);
            }
            x
        }
        pub fn pi() -> Number {
            let x = Number::new();
            unsafe {
                ca_pi(x.data, CALCIUM_CTX.ctx);
            }
            x
        }
        pub fn from_clong(i: c_long) -> Number {
            let x = Number::new();
            unsafe {
                ca_set_si(x.data, i, CALCIUM_CTX.ctx);
            }
            x
        }
        pub fn from_ulong(u: c_ulong) -> Number {
            let x = Number::new();
            unsafe {
                ca_set_ui(x.data, u, CALCIUM_CTX.ctx);
            }
            x
        }
        pub fn from_f64(f: f64) -> Number {
            let x = Number::new();
            unsafe {
                ca_set_d(x.data, f, CALCIUM_CTX.ctx);
            }
            x
        }
        pub fn from_str(s: &str) -> Option<Number> {
            unsafe {
                let mut x = Box::<fmpq>::new_uninit();
                fmpq_init(x.as_mut_ptr());
                let x = Box::into_raw(x.assume_init());
                let cs = CString::new(s).unwrap();
                let e = fmpq_set_str(x, cs.as_ptr(), 10);
                let n = Number::new();
                ca_set_fmpq(n.data, x, CALCIUM_CTX.ctx);
                fmpq_clear(x);
                drop(Box::from_raw(x));
                if e == 0 {
                    Some(n)
                } else {
                    None // maybe use an error message in the future
                }
            }
        }
    }
    impl_op_ex!(+ |a: &Number, b: &Number| -> Number {
        let res = Number::new();
        unsafe { ca_add(res.data, a.data, b.data, CALCIUM_CTX.ctx); }
        res
    });
    impl_op_ex!(-|a: &Number, b: &Number| -> Number {
        let res = Number::new();
        unsafe {
            ca_sub(res.data, a.data, b.data, CALCIUM_CTX.ctx);
        }
        res
    });
    impl_op_ex!(*|a: &Number, b: &Number| -> Number {
        let res = Number::new();
        unsafe {
            ca_mul(res.data, a.data, b.data, CALCIUM_CTX.ctx);
        }
        res
    });
    impl_op_ex!(/ |a: &Number, b: &Number| -> Number {
        let res = Number::new();
        unsafe { ca_div(res.data, a.data, b.data, CALCIUM_CTX.ctx); }
        res
    });
    impl Into<usize> for Number {
        fn into(self) -> usize {
            (&self).into()
        }
    }
    impl Into<usize> for &Number {
        fn into(self) -> usize {
            unsafe {
                let mut x = Box::<fmpz>::new_uninit();
                fmpz_init(x.as_mut_ptr());
                let x = Box::into_raw(x.assume_init());
                let e = ca_get_fmpz(x, self.data, CALCIUM_CTX.ctx);
                if e == 0 {
                    fmpz_clear(x);
                    drop(Box::from_raw(x));
                    panic!("is not integer");
                }
                if fmpz_sgn(x) < 0 || fmpz_abs_fits_ui(x) == 0 {
                    panic!("does not fit into usize");
                }
                let ui = fmpz_get_ui(x);
                fmpz_clear(x);
                drop(Box::from_raw(x));
                ui as usize
            }
        }
    }
    impl Drop for Number {
        fn drop(&mut self) {
            unsafe {
                ca_clear(self.data, CALCIUM_CTX.ctx);
                drop(Box::from_raw(self.data));
            }
        }
    }
    impl Eq for Number {}
    impl PartialEq for Number {
        fn eq(&self, other: &Self) -> bool {
            unsafe {
                let truth = ca_check_equal(self.data, other.data, CALCIUM_CTX.ctx);
                match truth {
                    truth_t_T_TRUE => true,
                    truth_t_T_FALSE => false,
                    _ => panic!("incomparable"),
                }
            }
        }
    }
    impl Display for Number {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            unsafe {
                let s = ca_get_str(self.data, CALCIUM_CTX.ctx);
                let res = f.write_str(CStr::from_ptr(s).to_str().unwrap());
                flint_free(s as *mut c_void);
                res
            }
        }
    }
    impl Debug for Number {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            Display::fmt(&self, f)
        }
    }
}

#[derive(Debug, Trace, Finalize)]
enum Expression {
    Number(calcium::Number),
    Bool(bool),
    String(String),
    BinOp(Gc<Expression>, BinOp, Gc<Expression>),
    Ident(String),
    Let(Gc<Expression>, Gc<Expression>, Gc<Expression>),
    If(Gc<Expression>, Gc<Expression>, Gc<Expression>),
    Tuple(Vec<(Option<Gc<Expression>>, Gc<Expression>)>),
    List(Vec<Gc<Expression>>),
    Selector(Gc<Expression>, Gc<Expression>),
    Index(Gc<Expression>, Gc<Expression>),
    Ascribe(Gc<Expression>, Gc<Type>),
    Abs(Gc<Expression>, Gc<Type>, Gc<Expression>), // add ident for parameter so that it always looks it up in the context? is the param type necessary?
    Closure(Vec<String>, Gc<Expression>),
    App(Gc<Expression>, Gc<Expression>), // Reuse for constructors?
    Fix(Gc<Expression>),
    Case(Gc<Expression>, Vec<(String, String, Gc<Expression>)>), // Compile exhaustive patterns to this, since all we need is an eliminator.
    // TODO: Effects
}

#[derive(Debug, Trace, Finalize)]
enum Type {
    None,
    // TODO
    // tuples, lists, string, bool, number, abs, variant, id
}

impl Expression {
    fn is_val(&self) -> bool {
        todo!()
    }
    fn new_number(n: Number) -> Gc<Expression> {
        Expression::Number(n).into()
    }
    fn new_bool(b: bool) -> Gc<Expression> {
        Expression::Bool(b).into()
    }
    fn new_string(s: &str) -> Gc<Expression> {
        Expression::String(String::from(s)).into()
    }
    fn new_binop(l: Gc<Expression>, op: BinOp, r: Gc<Expression>) -> Gc<Expression> {
        Expression::BinOp(l, op, r).into()
    }
    fn new_ident(s: &str) -> Gc<Expression> {
        Expression::Ident(String::from(s)).into()
    }
    fn new_let(name: Gc<Expression>, v: Gc<Expression>, x: Gc<Expression>) -> Gc<Expression> {
        Expression::Let(name, v, x).into()
    }
    fn new_if(cond: Gc<Expression>, then: Gc<Expression>, els: Gc<Expression>) -> Gc<Expression> {
        Expression::If(cond, then, els).into()
    }
    fn new_tuple(v: Vec<(Option<Gc<Expression>>, Gc<Expression>)>) -> Gc<Expression> {
        Expression::Tuple(v).into()
    }
    fn new_list(v: Vec<Gc<Expression>>) -> Gc<Expression> {
        Expression::List(v).into()
    }
    fn new_selector(x: Gc<Expression>, field: Gc<Expression>) -> Gc<Expression> {
        Expression::Selector(x, field).into()
    }
    fn new_index(x: Gc<Expression>, i: Gc<Expression>) -> Gc<Expression> {
        Expression::Index(x, i).into()
    }
    fn new_abs(bind: Gc<Expression>, ty: Gc<Type>, body: Gc<Expression>) -> Gc<Expression> {
        Expression::Abs(bind, ty, body).into()
    }
    fn new_app(fun: Gc<Expression>, arg: Gc<Expression>) -> Gc<Expression> {
        Expression::App(fun, arg).into()
    }
    fn new_closure(
        env: Vec<String>,
        lam: Gc<Expression>,
    ) -> Gc<Expression> {
        Expression::Closure(env, lam).into()
    }
    fn new_fix(x: Gc<Expression>) -> Gc<Expression> {
        Expression::Fix(x).into()
    }
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    // Assign,
}

simple_empty_finalize_trace![BinOp];

// struct Interpreter {
//     context: HashMap<String, >
// }

// struct Environment {
//     context: HashMap<String, Option<Gc<Expression>>>,
//     parent: Option<Gc<Environment>>,
// }

// impl Environment {
//     fn lookup(&self, name: &String) -> Gc<Expression> {
//         match &self.context[name] {
//             Some(v) => v.clone(),
//             None => match &self.parent {
//                 Some(p) => p.lookup(name),
//                 None => panic!("couldn't find identifier"),
//             },
//         }
//     }
//     fn insert(&mut self, name: String, x: Gc<Expression>) {
//         self.context.insert(name, Some(x));
//         // self.context[name] = Some(x);
//     }
// }

// struct ContextEntry {
//     name: String,
//     x: Gc<Expression>,
// }

#[derive(Trace, Finalize)]
struct Context {
    // stored at front (prepended)
    parent: Option<Gc<Context>>,
    name: String,
    x: Gc<Expression>,
}

impl Context {
    fn lookup(self: Gc<Context>, name: &String) -> Gc<Expression> {
        if &self.name == name {
            self.x.clone()
        } else if let Some(p) = self.parent.clone() {
            p.lookup(name)
        } else {
            panic!("couldn't find identifier")
        }
    }
    fn with(self: Gc<Context>, name: String, x: &Gc<Expression>) -> Gc<Context> {
        Context::new(Some(self), name, x)
    }
    fn new(parent: Option<Gc<Context>>, name: String, x: &Gc<Expression>) -> Gc<Context> {
        Context {
            parent: parent,
            name: name,
            x: x.clone(),
        }
        .into()
    }
}

enum Frame {
    Rv(Option<Gc<Expression>>),
    Rec {expr: Gc<Expression>, locals: RefCell<HashMap<String, Gc<Expression>>>, rv: Option<Gc<Expression>>},
}

impl Frame {
    fn new_rec(x: Gc<Expression>) -> Frame {
        Frame::Rec { expr: x, locals: RefCell::new(HashMap::new()), rv: None }
    }
    fn new_rec_hm(x: Gc<Expression>, locals: HashMap<String, Gc<Expression>>) -> Frame {
        Frame::Rec { expr: x, locals: RefCell::new(locals), rv: None }
    }
    fn new_rec_locals<const N: usize>(x: Gc<Expression>, locals: [(String, Gc<Expression>); N]) -> Frame {
        Frame::Rec { expr: x, locals: RefCell::new(HashMap::from(locals)), rv: None }
    }
}

fn set_rv(stack: &mut Vec<Frame>, v: Gc<Expression>) {
    stack.pop();
    match stack.last_mut() {
        Some(Frame::Rv(opt)) => opt.insert(v),
        Some(Frame::Rec { expr, locals, rv }) => rv.insert(v),
        None => panic!("expected frame to insert return value"),
    };
}

fn lookup_id(stack: &Vec<Frame>, s: &str) -> Gc<Expression> {
    for frame in stack.iter().rev() {
        if let Frame::Rec { expr: _, locals, rv: _ } = frame {
            if let Some(x) = locals.borrow().get(s) {
                return x.clone()
            }
        }
    }
    panic!("variable {s} was not in scope")
}

// given some expression, and the current stack, append to the map pairs of the free variables and their values from the stack
// fn append_free_vars(env: &mut HashMap<String, Gc<Expression>>, stack: &Vec<Frame>, expr: &Gc<Expression>) {
//     // mat
// }

fn resolve_names<'a>(m: &mut Vec<&'a str>, free_vars: &mut Vec<String>, dist: usize, prog: &'a Gc<Expression>) -> Gc<Expression> {
    match &**prog {
        Expression::BinOp(l, op, r) => {
            let l = resolve_names(m, free_vars, dist, l);
            let r = resolve_names(m, free_vars, dist, r);
            Expression::new_binop(l, *op, r)
        },
        Expression::Let(bind, rhs, body) => {
            let rhs = resolve_names(m, free_vars, dist, rhs);
            let Expression::Ident(x) = &**bind else { panic!("expected identifier in let binding") };
            m.push(x);
            let body = resolve_names(m, free_vars, dist+1, body);
            m.pop(); // pop here or inside?
            Expression::new_let(bind.clone(), rhs, body)
        },
        Expression::If(cond, then, els) => {
            let cond = resolve_names(m, free_vars, dist, cond);
            let then = resolve_names(m, free_vars, dist,  then);
            let els = resolve_names(m, free_vars, dist,  els);
            Expression::new_if(cond, then, els)
        },
        Expression::Tuple(_) => todo!(),
        Expression::List(_) => todo!(),
        Expression::Selector(_, _) => todo!(),
        Expression::Index(_, _) => todo!(),
        Expression::Ascribe(_, _) => todo!(),
        Expression::Ident(x) => {
            // look up x in m up to dist
            // if not found, then it's a free variable
            // append to free_vars
            let mut i = 0;
            let mut found = false;
            while i <= dist {
                let var = m[m.len() - i - 1];
                if var == x {
                    found = true;
                    break;
                }
            }
            if !found {
                free_vars.push(x.to_string());
            }
            prog.clone()
        },
        Expression::Abs(bind, /*TODO: resolve names in types */ ty, body) => {
            let Expression::Ident(x) = &**bind else { panic!("expected identifier in lambda binding") };
            m.push(x);
            let mut free_vars = Vec::new();
            let body = resolve_names(m, &mut free_vars, 0, body);
            m.pop();
            Expression::new_closure(free_vars, Expression::new_abs(bind.clone(), ty.clone(), body))
        },
        Expression::App(fun, arg) => {
            let fun = resolve_names(m, free_vars, dist, fun);
            let arg = resolve_names(m, free_vars, dist, arg);
            Expression::new_app(fun, arg)
        },
        Expression::Fix(x) => Expression::new_fix(resolve_names(m, free_vars, dist, x)),
        Expression::Case(_, _) => todo!(),
        _ => prog.clone(),
    }
}

fn eval_with_stack(mut stack: Vec<Frame>) -> Gc<Expression> {
    while let Some(frame) = stack.last() {
        let mut to_set : Option<Gc<Expression>> = None;
        let mut to_push : Option<Frame> = None;
        match frame {
            Frame::Rv(Some(rv)) => return rv.clone(),
            Frame::Rv(None) => panic!("expected return value at the top of the stack"),
            Frame::Rec { expr, locals, rv } => match &**expr {
                Expression::Number(_) => to_set = Some(expr.clone()),
                Expression::Bool(_) => to_set = Some(expr.clone()),
                Expression::String(_) => to_set = Some(expr.clone()),
                Expression::Closure(env, abs) => to_set = Some(expr.clone()),
                Expression::Ident(s) => {
                    let val = lookup_id(&stack, s);
                    to_set = Some(val)
                },
                Expression::BinOp(l, op, r) => match rv {
                    Some(v) => {
                        let mut locals = locals.borrow_mut();
                        if let Some(v1) = locals.get("1") {
                            let res = match (&**v1, op, &**v) {
                                (Expression::Number(l), BinOp::Add, Expression::Number(r)) => Expression::new_number(l+r),
                                (Expression::Number(l), BinOp::Sub, Expression::Number(r)) => Expression::new_number(l-r),
                                (Expression::Number(l), BinOp::Mul, Expression::Number(r)) => Expression::new_number(l*r),
                                (Expression::Number(l), BinOp::Div, Expression::Number(r)) => Expression::new_number(l/r),
                                _ => panic!("left and right were not numbers"),
                            };
                            to_set = Some(res);
                        } else {
                            locals.insert("1".to_string(), v.clone());
                            to_push = Some(Frame::new_rec(r.clone()));
                        }
                    },
                    None => to_push = Some(Frame::new_rec(l.clone())),
                },
                Expression::Let(id, x, body) if let Expression::Ident(id) = &**id => match rv {
                    Some(v) => {
                        let mut locals = locals.borrow_mut();
                        if locals.contains_key("1") {
                            to_set = Some(v.clone());
                        } else {
                            locals.insert("1".to_string(), v.clone());
                            to_push = Some(Frame::new_rec_locals(body.clone(), [(id.clone(), v.clone())]));
                        }
                    },
                    None => to_push = Some(Frame::new_rec(x.clone())),
                },
                Expression::Let(_, _, _) => todo!(),
                Expression::If(cond, then, els) => match rv {
                    Some(v) => {
                        let mut locals = locals.borrow_mut();
                        if locals.contains_key("1") {
                            to_set = Some(v.clone());
                        } else {
                            locals.insert("1".to_string(), v.clone());
                            let Expression::Bool(b) = **v else { panic!("expected boolean") };
                            to_push = if b {
                                Some(Frame::new_rec(then.clone()))
                            } else {
                                Some(Frame::new_rec(els.clone()))
                            }
                        }
                    },
                    None => to_push = Some(Frame::new_rec(cond.clone())),
                },
                Expression::Tuple(_) => todo!(),
                Expression::List(_) => todo!(),
                Expression::Selector(_, _) => todo!(),
                Expression::Index(_, _) => todo!(),
                Expression::Ascribe(_, _) => todo!(),
                Expression::Abs(_, _, _) => todo!(), // subsumed by closure
                Expression::App(fun, arg) => match rv {
                    Some(rv) => {
                        let mut locals = locals.borrow_mut();
                        if locals.contains_key("2") {
                            // return body
                            to_set = Some(rv.clone());
                        } else if let Some(v1) = locals.remove("1") {
                            // eval body
                            let Expression::Closure(env, abs) = &*v1 else { panic!("expected closure") };
                            let Expression::Abs(bind, _, body) = &**abs else { panic!("expected abs") };
                            let Expression::Ident(bind) = &**bind else { panic!("expected ident") };
                            locals.insert("2".to_string(), rv.clone());
                            let mut fn_locals = HashMap::new();
                            for id in env {
                                fn_locals.insert(id.to_string(), lookup_id(&stack, id));
                            }
                            fn_locals.insert(bind.to_string(), rv.clone());
                            to_push = Some(Frame::new_rec_hm(body.clone(), fn_locals));
                        } else {
                            // eval arg
                            locals.insert("1".to_string(), rv.clone());
                            to_push = Some(Frame::new_rec(arg.clone()));
                        }
                    },
                    // eval fun
                    None => to_push = Some(Frame::new_rec(fun.clone())),
                },
                Expression::Fix(x) => match rv {
                    Some(rv) => {
                        let mut locals = locals.borrow_mut();
                        if locals.contains_key("1") {
                            to_set = Some(rv.clone());
                        } else {
                            locals.insert("1".to_string(), rv.clone());
                            let Expression::Closure(env, abs) = &**rv else { panic!("expected closure") };
                            let Expression::Abs(bind, _, body) = &**abs else { panic!("expected abs") };
                            let Expression::Ident(bind) = &**bind else { panic!("expected ident") };
                            let mut fn_locals = HashMap::new();
                            for id in env {
                                fn_locals.insert(id.to_string(), lookup_id(&stack, id));
                            }
                            let arg = Expression::new_fix(rv.clone());
                            fn_locals.insert(bind.to_string(), arg);
                            to_push = Some(Frame::new_rec_hm(body.clone(), fn_locals));
                        }
                    },
                    None => to_push = Some(Frame::new_rec(x.clone())),
                },
                Expression::Case(_, _) => todo!(),
            },
        }
        if let Some(x) = to_set {
            set_rv(&mut stack, x);
        }
        if let Some(f) = to_push {
            stack.push(f);
        }
    }
    unreachable!()
}

fn eval_stack(x: Gc<Expression>) -> Gc<Expression> {
    eval_with_stack(vec![
        Frame::Rv(None),
        Frame::new_rec(x),
    ])
}

fn eval(ctx: &Option<Gc<Context>>, x: &Gc<Expression>) -> Gc<Expression> {
    let (ctx, x) = (ctx.clone(), x.clone());
    match &*x {
        Expression::Number(_) => x,
        Expression::Bool(_) => x,
        Expression::String(_) => x,
        Expression::BinOp(left, op, right) => match (&*eval(&ctx, left), op, &*eval(&ctx, right)) {
            (Expression::Number(l), BinOp::Add, Expression::Number(r)) => Expression::new_number(l+r),
            (Expression::Number(l), BinOp::Sub, Expression::Number(r)) => Expression::new_number(l-r),
            (Expression::Number(l), BinOp::Mul, Expression::Number(r)) => Expression::new_number(l*r),
            (Expression::Number(l), BinOp::Div, Expression::Number(r)) => Expression::new_number(l/r),
            _ => panic!("left and right were not numbers")
        },
        Expression::Ident(id) if let Some(ctx) = ctx => ctx.lookup(id),
        Expression::Let(id, v, x) if let Expression::Ident(id) = &**id => match ctx.clone() {
            Some(ctx) => eval(&Some(ctx.with(id.to_string(), v)), x),
            None => eval(&Some(Context::new(None, id.to_string(), v)), x),
        },
        Expression::If(cond, then, els) if let Expression::Bool(cond) = *eval(&ctx, cond) =>
            if cond { eval(&ctx, then) } else { eval(&ctx, els) },
        Expression::Tuple(elements) => Expression::new_tuple(elements.iter().map(
            |elem| (elem.0.clone(), eval(&ctx, &elem.1))
        ).collect()),
        Expression::List(elements) => Expression::new_list(elements.iter().map(|elem| eval(&ctx, elem)).collect()),
        Expression::Selector(x, sel) if let Expression::Ident(id) = &**sel =>
        if let Expression::Tuple(r) = &*eval(&ctx, x) {
            r.iter().find_map(|ele|
                if let Some(x) = &ele.0 && let Expression::Ident(got)= &**x && id == got {
                    Some(ele.1.clone())
                } else {
                    None
                }).unwrap()
        } else {
            panic!("selector expression wasn't made up of tuple and ident")
        },
        Expression::Index(x, i) if let Expression::Number(i) = &*eval(&ctx, i) => match &*eval(&ctx, x) {
            Expression::Tuple(r) => r[Into::<usize>::into(i)].1.clone(),
            Expression::List(l) => l[Into::<usize>::into(i)].clone(),
            _ => panic!("expression is not indexable"),
        }
        Expression::Abs(_, _, _) => x,
        Expression::App(f, arg) => match &*eval(&ctx, f) {
            Expression::Abs(param, _param_ty, body) => {
                let arg = eval(&ctx, arg);
                if !arg.is_val() {
                    todo!()
                }
                if let Expression::Ident(id) = &**param {
                    match ctx.clone() {
                        Some(ctx) => eval(&Some(ctx.with(id.to_string(), &arg)), body),
                        None => eval(&Some(Context::new(None, id.to_string(), &arg)), body),
                    }
                } else {
                    todo!()
                }
            },
            _ => todo!(),
            // TODO: handle constructor. is it an ident we look up in some scope?
        },
        Expression::Fix(x)  if let Expression::Abs(param, _, body) = &*eval(&ctx, x) =>
        if let Expression::Ident(id) = &**param {
            match ctx.clone() {
                Some(ctx) => eval(&Some(ctx.with(id.to_string(), body)), body),
                None => eval(&Some(Context::new(None, id.to_string(), body)), body),
            }
        } else {
            todo!()
        },
        Expression::Ascribe(x, _) => x.clone(),
        _ => panic!("unexpected expression"),
    }
}

fn main() {
    // let prog = Expression::new_let(
    //     Expression::new_ident("x"),
    //     Expression::new_number(Number::from_ulong(3)),
    //     Expression::new_binop(
    //         Expression::new_number(Number::one()),
    //         BinOp::Add,
    //         Expression::new_ident("x"),
    //     ),
    // );
    // let x = Expression::new_binop(Expression::new_number(Number::one()), BinOp::Add, Expression::new_number(Number::from_ulong(2)));
    // let prog = Expression::new_if(Expression::new_let(Expression::new_ident("x"), Expression::new_bool(false), Expression::new_ident("x")), Expression::new_number(Number::from_ulong(3)), Expression::new_number(Number::from_ulong(4)));
    let prog = Expression::new_app(
        Expression::new_abs(
            Expression::new_ident("x"),
            Type::None.into(), // don't care about type
            Expression::new_ident("x"),
        ),
        Expression::new_number(Number::from_ulong(3)),
    );
    let mut m = Vec::new();
    let mut free_vars = Vec::new();
    let prog = resolve_names(&mut m, &mut free_vars, 0, &prog);
    let stackres = eval_stack(prog);
    println!("stack res = {:?}", stackres);
    // let prog = Expression::new_selector(
    //     Expression::new_tuple(vec![(
    //         Some(Expression::new_ident("x")),
    //         Expression::new_number(2),
    //     )]),
    //     Expression::new_ident("x"),
    // );
    // let prog = Expression::new_index(
    //     Expression::new_tuple(vec![(
    //         Some(Expression::new_ident("x")),
    //         Expression::new_number(2),
    //     )]),
    //     Expression::new_number(0),
    // );
    // let prog : Gc<_> = Expression::If(
    //     Expression::Bool(false).into(),
    //     Expression::Number(3).into(),
    //     Expression::BinOp(
    //         Expression::Number(1).into(),
    //         BinOp::Add,
    //         Expression::Number(1).into(),
    //     ).into(),
    // ).into();
    // let prog : Gc<_> = Expression::Tuple(
    //     vec![
    //         (None, Expression::Number(1).into()),
    //         (Some(Expression::Ident(String::from("x").into()).into()),
    //             Expression::BinOp(
    //             Expression::Number(1).into(),
    //             BinOp::Add,
    //             Expression::Number(1).into(),
    //         ).into()),
    //     ],
    // ).into();
    // let ctx = None;
    // let env: Gc<_> = Environment{
    //     context: HashMap::new(),
    //     parent: None,
    // }.into();
    // let res = eval(&ctx, &prog);
    // println!("res={:?}", res);
    // println!()
    // unsafe {
    //     let mut ctx: ca_ctx_struct = mem::zeroed();
    //     let mut x: ca_struct = mem::zeroed();
    //     ca_ctx_init(&mut ctx);
    //     ca_init(&mut x, &mut ctx);
    //     ca_pi(&mut x, &mut ctx);
    //     ca_sub_ui(&mut x, &mut x, 3, &mut ctx);
    //     ca_pow_ui(&mut x, &mut x, 2, &mut ctx);
    //     ca_print(&mut x, &mut ctx);
    //     println!();
    //     println!("Computed with calcium-{}", CStr::from_ptr(calcium_version()).to_str().unwrap());
    //     ca_clear(&mut x, &mut ctx);
    //     ca_ctx_clear(&mut ctx);
    //     flint_cleanup();
    // }
    // let one = Number::one();
    // let pi = Number::pi();
    // let one = Number::from_f64(0.1);
    // let two = Number::from_f64(0.2);
    // let a = Number::from_ulong(1) / Number::from_ulong(10);
    // let b = Number::from_ulong(2) / Number::from_ulong(10);
    // let c = Number::from_ulong(3) / Number::from_ulong(10);
    // println!("a={}", a);
    // println!("b={}", b);
    // println!("c={}", c);
    // println!("a+b==c:{}", a+b == c);
    // let x = Number::from_str("1/2");
    // println!("x={}", x.unwrap());
    // let y : usize = Number::from_ulong(1).into();
    // println!("y={}", y);
    // unsafe {
    // let mut ctx = MaybeUninit::<ca_ctx_struct>::uninit();
    // ca_ctx_init(ctx.as_mut_ptr());
    // let ctx = &mut ctx.assume_init() as *mut ca_ctx_struct;

    // let mut ca = MaybeUninit::<ca_struct>::uninit();
    // ca_init(ca.as_mut_ptr(), ctx);
    // ca_one(ca.as_mut_ptr(), ctx);
    // let ca = &mut ca.assume_init() as *mut ca_struct;
    // let pctx = ctx.as_mut_ptr();
    // let mut x: MaybeUninit<ca_struct> = MaybeUninit::uninit();
    // let px = x.as_mut_ptr();
    // ca_ctx_init(pctx);
    // ctx.assume_init();
    // let mut c = ctx.assume_init();
    // let p: *mut ca_ctx_struct = &mut c as *mut ca_ctx_struct;
    // let px : *mut ca_struct = x.as_mut_ptr();
    // ca_one(px, pctx);
    // x.assume_init();
    // let one = &mut x.assume_init();
    // }
}
