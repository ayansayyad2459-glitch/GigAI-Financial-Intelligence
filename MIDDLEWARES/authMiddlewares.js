    const jwt = require('jsonwebtoken')
    module.exports  = async (req, res , next)=>{
        const token = req.header('x-auth-token');
        if(!token){
            return res.status(401).json({message: "No token, authorization denied"});

        }
        // Verify token
        try{
             const decodeed = jwt.verify(token, process.env.JWT_SECRET);

             req.user = decodeed.user;
            next();
        } catch(err){
            res.status(401).json({message: "Token is not valid"});
        }

        
    }
// what things to add in this file
// 1. Import the jwt library to handle JSON Web Tokens.
// 2. Create a middleware function that checks for the presence of a token in the request header.
// 3. If the token is not present, return a 401 Unauthorized response.
// 4. If the token is present, verify it using the jwt.verify method and the secret key from the environment variables.
// 5. If the token is valid, decode it and attach the user information to the request object for use in subsequent middleware or route handlers.
// 6. If the token is invalid, return a 401 Unauthorized response.
// This middleware can be used to protect routes that require authentication by ensuring that only requests with a valid token can access those routes.